import evaluate
import torch
from whisper.decoding import DecodingOptions
from utils import decode_labels,strip_timestamps



def testing_loop(model, test_loader, device, compute_metrics=True):
    """
    Real Whisper ASR evaluation (true autoregressive decoding).
    """
    model.to(device)
    model.eval()

    all_predictions = []
    all_references = []

    if compute_metrics:
        cer_metric = evaluate.load("cer")
        wer_metric = evaluate.load("wer")

    decode_options = DecodingOptions(
        language="el",
        task="transcribe",
        without_timestamps=True,
        beam_size=5,
        patience=1.0,
    )

    with torch.no_grad():
        for batch in test_loader:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"]

            audio_features = model.model.encoder(input_features)


            decoded_texts = []
            for i in range(audio_features.shape[0]):
                result = model.model.decode(
                    audio_features[i : i + 1],
                    decode_options,
                )
                decoded_texts.append(result[0].text)

            ref_texts = decode_labels(model.tokenizer, labels)
            ref_texts = [strip_timestamps(r) for r in ref_texts]
            all_predictions.extend(decoded_texts)
            all_references.extend(ref_texts)

    results = {
        "predictions": all_predictions,
        "references": all_references,
    }

    if compute_metrics:
        results["cer"] = cer_metric.compute(
            predictions=all_predictions,
            references=all_references,
        )
        results["wer"] = wer_metric.compute(
            predictions=all_predictions,
            references=all_references,
        )

    return results
