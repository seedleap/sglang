# LingBot 20-case first-frame test set

This package contains 20 visually reviewed first-person cases sampled from the
July 2026 LingBot production image batches.

## Composition

- 5 human viewpoints with hands or forearms visible.
- 5 human viewpoints with the environment filling the frame.
- 5 nonhuman organic viewpoints.
- 5 nonhuman mechanical viewpoints.
- Action families: WASD 8, IJKL 7, mixed 5.
- Visual styles: realistic 10, stylized 10.
- Every first frame is 1672 x 941 PNG and passed visual review.

## Prompt policy

Each prompt was rebuilt as a concise positive video condition from the selected
subject viewpoint, scene description, visually verified viewpoint anchor, style,
and exact action segments. Source image-generation prompts were not copied.
The package excludes generation_prompt, clean_prompt, negative_prompt, and
video_prompt_override from every delivered prompt and metadata file.

## Files

- images/: original full-resolution first frames downloaded from S3.
- prompts/: one positive prompt per case.
- metadata/: complete case metadata plus the 129-frame action condition.
- manifest.jsonl: compact machine-readable manifest.
- manifest.csv: spreadsheet-friendly manifest.
- summary.json: dataset composition and prompt policy.
- contact_sheet.jpg: visual review sheet in case order.
- SHA256SUMS: checksums for all delivered artifacts.

LingBot World 2 experiments and derivatives are treated as non-commercial under
CC BY-NC-SA 4.0. Retain attribution and share-alike obligations when redistributing.
