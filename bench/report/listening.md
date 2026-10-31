# Listening protocol

The quality harness measures signal properties; it cannot measure whether speech sounds like
speech. This protocol produces the one number it deliberately leaves as `"pending"`.

## Stimuli

`audio/` holds one 16-bit mono WAV per case per dtype, named
`<device>-<dtype>-<case>.wav`. They were produced at `temperature: 0`, so replaying the sweep
yields byte-identical files.

## Procedure

1. Blind the file names: replace each `-fp32-`, `-q4-`, `-q4f16-` segment with `-A-`, `-B-`, `-C-`
   using a shuffled mapping you keep out of sight until scoring is done.
2. Have at least three raters listen on headphones, in one sitting, in the same order.
3. For each case, rate naturalness on a 1–5 scale (1 = unintelligible, 3 = understandable but
   clearly synthetic, 5 = indistinguishable from a recording).
4. Record the median per case and per dtype, then the mean of the per-case medians.
5. Enter the result as `listening.score` in `quality-node-cpu-<dtype>.json`, and set
   `listening.raters` to the number of raters. Do not change anything else in the report.

## Reporting

Quote the score only with its case count, rater count, and hardware, exactly like the latency
tables: a MOS without that context is a number someone made up.
