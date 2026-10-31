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
| Phonon 31M EN (`onnx-community/phonon-31m-en-ONNX`)         | Apache-2.0           | Initial English MVP candidate; ONNX export published, Transformers.js integration still to be validated in M1.    |
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
MIT-licensed and bundles ONNX Runtime (`onnxruntime-web`, MIT). Their licenses
apply to the bundled code in the engine's distribution; see the respective
package licenses for details.
