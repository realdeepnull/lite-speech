# NOTICE

## Code license

The `lite-speech` packages (`@lite-speech/core`, `@lite-speech/engine-onnx`)
are licensed under the MIT License. See [LICENSE](LICENSE).

## Model assets are not shipped

No model weights, voices, tokenizer files, or ONNX graphs are published inside
the npm packages. Model and voice assets are downloaded from pinned, revisioned
sources at provisioning time, and every manifest entry records the license of
the model it points to.

A model's license is **not** the library's license. Applications are responsible
for complying with the model license they choose to use.

## Model families and licenses

| Model family                                                | License              | Notes                                                                                                             |
| ----------------------------------------------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Phonon 31M EN (`onnx-community/phonon-31m-en-ONNX`)         | Apache-2.0           | English MVP profile. ONNX export and the low-level Transformers.js path were verified in M1.                      |
| Phonon 40M DE (`gradium/phonon-40m-de`)                     | CC BY-NC 4.0 (gated) | Non-commercial, gated, and without a verified ONNX/Transformers.js export at the reference date. **Not shipped.** |
| Phonon 40M multilingual (`gradium/phonon-40m-multilingual`) | CC BY-NC 4.0 (gated) | Non-commercial and gated. **Not shipped.**                                                                        |

Rules this repository follows:

- Gated or non-commercial weights are never bundled or redistributed.
- Every shipped model profile documents its license, measured download size, and
  streaming class.
- No model substitution happens silently: a different model, voice, dtype, or
  backend is always surfaced through an event or a capability report.
- Commercial use of a `CC BY-NC 4.0` model family is not granted by the MIT
  license of this library and must be evaluated independently.

## Third-party runtime dependencies

`@lite-speech/engine-onnx` depends on `@huggingface/transformers`, which is
Apache-2.0 licensed and bundles ONNX Runtime (`onnxruntime-node` /
`onnxruntime-web`, MIT) plus `@huggingface/tokenizers` (Apache-2.0). Their
licenses apply to the bundled code in the engine's distribution; see the
respective package licenses for details.

## Model and voice licenses used by the spike

The M1 spike (`spike/`, never published) resolves
`onnx-community/phonon-31m-en-ONNX` at commit
`4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4` and its default voice from
`gradium/phonon-voices` at `0ee17205f254b5abc43b266c8a0f215565c352f7`; both are
Apache-2.0. See
[docs/spike/transformers-js-phonon.md](docs/spike/transformers-js-phonon.md).
