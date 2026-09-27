Files in this directory (ast_checker.py, enums.py, type_mappings.py,
type_convertor/java_type_converter.py, type_convertor/js_type_converter.py) are vendored,
with import-path adjustments and one functional change (see below), from:

https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard

Copyright the Gorilla / Berkeley Function Calling Leaderboard authors.
Licensed under the Apache License, Version 2.0.
http://www.apache.org/licenses/LICENSE-2.0

Reference: Patil, S. G., et al. The Berkeley Function Calling Leaderboard (BFCL): From Tool
Use to Agentic Evaluation of Large Language Models. OpenReview, 2025.

Scope decision: this project's BFCL-Hi evaluation is restricted to the BFCL_v2_live_simple
category (single function call per item) from nvidia/BFCL-Hi. The multi-function categories
(multiple, parallel, parallel_multiple) and the relevance/irrelevance categories (which test
whether a model correctly abstains from calling any function, a fundamentally different check
than AST matching, and which nvidia/BFCL-Hi does not ship possible-answer ground truth for
in the same format) were excluded due to implementation scope. This is a disclosed scoping
decision, not an evaluation-fidelity compromise on the category that is used.

Functional change from upstream: convert_func_name() originally looked up the calling model in
MODEL_CONFIG_MAPPING, a large table of Gorilla's own explicitly-supported proprietary model
identifiers, to decide whether to convert dots to underscores in function names (a workaround
for providers, e.g. OpenAI, that don't permit dots in function names). None of this project's
six evaluation-panel models are keys in that table, and none of them are called through a
proprietary function-calling API layer that would impose this constraint — each model is
prompted directly with the function name as given in the dataset's function schema. This
function was therefore replaced with an identity pass-through (returns the function name
unchanged) rather than vendoring the full, large, and here-irrelevant MODEL_CONFIG_MAPPING
table. All other AST-checking logic (function name matching, required-parameter checking,
per-parameter type and value checking, list/dict structural matching) is unmodified from
upstream.

Known limitation, not patched (a property of official BFCL semantics, not a bug): the
official type_checker enforces strict type matching between a model's argument value and
the ground truth's expected type. A model that outputs a quoted numeric ID (e.g. "7890")
where the ground truth is an unquoted integer (7890) is scored as incorrect, even though
the represented value is identical. This is standard, unmodified BFCL behavior — the same
strictness applies in the original English leaderboard — and is left as-is to preserve
comparability with published BFCL results, rather than adding our own type-coercion leniency
that would constitute a deviation from official methodology. This is disclosed here because
it was empirically observed to affect at least one evaluated model (ai4bharat/Airavata) in
this study, which showed a tendency to quote numeric argument values.