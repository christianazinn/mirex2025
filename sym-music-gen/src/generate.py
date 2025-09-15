import argparse
import tqdm
import torch
import symusic
import miditok
import transformers
import json

from pathlib import Path


def __parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=Path)
    parser.add_argument("prompt", type=Path)
    parser.add_argument("--num-generations", type=int, default=4)

    return parser.parse_args()


class BarStoppingCriteria(transformers.StoppingCriteria):
    def __init__(self, num_bars: int, bar_delimiter_token_id: int):
        super().__init__()
        self._num_bars = num_bars
        self._bar_delimiter_token_id = bar_delimiter_token_id

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs):
        return (input_ids == self._bar_delimiter_token_id).sum(dim=1) >= self._num_bars


def notes_to_score(notes):
    score = symusic.Score(ttype=symusic.TimeUnit.quarter)

    track = symusic.Track(ttype=symusic.TimeUnit.quarter)

    for n in notes:
        # In symusic the note duration is specified in quarters.
        # In the MIREX prompt it is specified in sixteenth notes
        note = symusic.Note(
            time=n["start"] / 4,
            duration=n["duration"] / 4,
            pitch=n["pitch"],
            velocity=90,
            ttype=symusic.TimeUnit.quarter,
        )
        track.notes.append(note)

    score.tracks.append(track)

    score = score.to("tick")

    return score


# def custom_decoding(model, prompt_tokens: torch.tensor, num_bars: int = 16,


def main():
    args = __parse_args()

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model).eval().cuda()

    tokenizer = miditok.REMI.from_pretrained(args.model)

    with args.prompt.open() as io:
        prompt = json.load(io)

    prompt_score = notes_to_score(prompt["prompt"])

    prompt_tokens = tokenizer.encode(prompt_score)[0].ids

    prompt_tokens_pt = torch.tensor(prompt_tokens).unsqueeze(0).cuda()
    attention_mask = torch.ones_like(prompt_tokens_pt).cuda()

    output_dir = Path(f"{args.prompt.stem}_generations")
    output_dir.mkdir(exist_ok=True)
    prompt_score.dump_midi(str(output_dir / "prompt.mid"))

    bar_stopping_criteria = BarStoppingCriteria(
        num_bars=16, bar_delimiter_token_id=tokenizer.vocab["Bar_None"]
    )

    for n in tqdm.tqdm(range(1, args.num_generations + 1)):
        with torch.no_grad():
            output = model.generate(
                prompt_tokens_pt,
                max_new_tokens=3000,
                attention_mask=attention_mask,
                stopping_criteria=[bar_stopping_criteria],
                pad_token_id=tokenizer.vocab["PAD_None"],
                temperature=1.5,
                top_p=0.90,
                do_sample=True,
                num_beams=3,
                repetition_penalty=0.5,
            )
        full_score = tokenizer.decode(output.cpu())

        full_score.dump_midi(str(output_dir / f"sample_{n:02d}.mid"))


if __name__ == "__main__":
    main()
