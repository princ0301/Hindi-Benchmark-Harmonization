Files in this directory (instructions.py, instructions_registry.py, instructions_util.py,
evaluation_lib.py) are vendored from:

https://github.com/google-research/google-research/tree/master/instruction_following_eval

Copyright 2026 The Google Research Authors.
Licensed under the Apache License, Version 2.0.
http://www.apache.org/licenses/LICENSE-2.0

Original paper: Zhou, J., Lu, T., Mishra, S., Brahma, S., Basu, S., Luan, Y., Zhou, D., & Hou, L.
(2023). Instruction-Following Evaluation for Large Language Models. arXiv:2311.07911.

Vendored rather than reimplemented to preserve exact scoring-logic fidelity with the
official IFEval methodology, per the project's evaluation harness design.

Known deviation from upstream: instructions.py's LetterFrequencyChecker originally validated
the target letter against the ASCII a-z range only (`ord(letter.lower()) < 97 or > 122`),
silently substituting a random English letter whenever a non-ASCII letter (e.g. Devanagari)
was supplied — this would have made Hindi letter-frequency checks (keywords:letter_frequency,
used directly in nvidia/IFEval-Hi with Devanagari letters such as "श") silently test a random
unrelated letter instead of the one specified in the prompt. This validation was patched to
use Python's Unicode-aware `str.isalpha()` instead of a hardcoded ASCII codepoint range,
preserving the original fallback behavior for genuinely invalid inputs (empty, multi-character,
non-letter) while correctly supporting Devanagari and other non-Latin scripts. All other
checker logic is unmodified from upstream.

Known deviation from upstream: instructions.py's ParagraphFirstWordCheck (length_constraints:
nth_paragraph_first_word) had two robustness gaps discovered when scoring real Hindi model
outputs against nvidia/IFEval-Hi:
  (a) build_description() never stripped whitespace from the target first_word value before
      storing it. Several items in nvidia/IFEval-Hi's own kwargs field carry unintentional
      trailing whitespace (e.g. "लोक ", "सती  " with two trailing spaces), which made an exact
      match against any real model output impossible regardless of compliance. Patched to
      `.strip()` the value, matching the treatment EndChecker already gives end_phrase.
  (b) The word-cleaning logic's punctuation-stripping set ({".", ",", "?", "!", "'", '"'})
      did not include markdown emphasis markers. A very common LLM formatting habit — bolding
      the opening word/phrase with "**" — caused the leading asterisks to be captured as part
      of the extracted "first word," corrupting the comparison for any model that uses this
      formatting style, independent of whether it satisfied the actual instruction. Patched to
      also strip "*", "_", and "#" from both ends of the extracted token. This is a
      general robustness fix (equally relevant to English IFEval, not Hindi-specific), applied
      because it was directly observed corrupting scores across every one of the six models in
      this study identically, in a case where all models were coincidentally using markdown-bold
      openers — see the paper's methodology/limitations discussion for the empirical evidence
      that motivated this fix.

Known limitation, not patched (a substantive linguistic finding, not a bug): even after the
above first_word fixes, ParagraphFirstWordCheck's exact-token-match design does not
accommodate Hindi's productive compound-word morphology. A response beginning with
"लोककथाएँ" ("folktales", a fluent, natural compound formed from "लोक" + "कथाएँ") is scored as
non-compliant when the required first_word is "लोक", even though a Hindi speaker would likely
consider this a reasonable interpretation of "start with लोक". This reflects a genuine mismatch
between an evaluation design built for English (where such instructions cleanly resolve to a
single space-delimited token) and Hindi's morphology, rather than an implementation bug, and is
intentionally left unpatched — "fixing" it would require subjective linguistic judgment about
what counts as a compliant compound rather than a mechanical correction.

Known limitation: instructions_util.py's count_sentences() uses NLTK's English-trained
Punkt sentence tokenizer, which does not recognize the Hindi purna viram ("।") as a
sentence boundary. This affects only the length_constraints:number_sentences instruction
type; other instruction types (word counts, keyword checks, format checks) are not
affected by this English-tokenizer dependency.