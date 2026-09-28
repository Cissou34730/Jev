import json
import os
import sys
from datetime import datetime
from time import perf_counter


def main():
    payload = json.load(sys.stdin)
    tickets = payload["tickets"]
    batch_size = int(payload["batchSize"])
    laya_batch_size = int(os.environ["LAYA_BATCH_SIZE"])
    torch_threads = int(os.environ["LAYA_TORCH_THREADS"])
    model = payload["model"]
    instruction = payload["instruction"].strip()

    import torch

    torch.set_num_threads(torch_threads)
    print(
        f"CPU inference settings: torch_threads={torch_threads}, "
        f"outer_ticket_batch_size={batch_size}, "
        f"laya_internal_batch_size={laya_batch_size}, "
        "sort_by_length=True",
        file=sys.stderr,
        flush=True,
    )

    from laya import Router

    router = Router(device="cpu")
    questions = {
        "automatable": {
            "type": "noul",
            "instructions": f"{instruction} Apply this only to the ticket description.",
            "criteria": {
                "false": "The ticket does not meet the instruction and cannot be automated.",
                "true": "The ticket meets the instruction and can be automated.",
            },
            "labels": {"false": "B", "true": "A"},
        }
    }

    batches = []
    checkpoints = set()
    total_batches = (len(tickets) + batch_size - 1) // batch_size
    for start in range(0, len(tickets), batch_size):
        batch = tickets[start : start + batch_size]
        requests = []
        for ticket in batch:
            request = {
                "state": {"description": ticket["description"]},
                "questions": questions,
            }
            if model != "auto":
                request["model"] = model
            requests.append(request)

        batch_number = start // batch_size + 1
        batches.append((batch_number, batch, requests))
        for request in requests:
            route = router.route(
                request["state"],
                request["questions"],
                model=request.get("model"),
            )
            checkpoints.add(route.model)

    checkpoints = sorted(checkpoints)
    print(
        f"Loading checkpoint(s) before timing: {', '.join(checkpoints)}",
        file=sys.stderr,
        flush=True,
    )
    router.preload(checkpoints)

    results = []
    inference_started_at = None
    inference_finished_at = None
    inference_seconds = 0.0
    for batch_number, batch, requests in batches:
        try:
            call_started_at = datetime.now().astimezone()
            call_started = perf_counter()
            if inference_started_at is None:
                inference_started_at = call_started_at
            try:
                predictions = router.predict_batch(
                    requests,
                    batch_size=laya_batch_size,
                    sort_by_length=True,
                )
            finally:
                inference_seconds += perf_counter() - call_started
                inference_finished_at = datetime.now().astimezone()
            if len(predictions) != len(batch):
                raise RuntimeError(
                    f"Laya returned {len(predictions)} answers for {len(batch)} tickets."
                )

            for ticket, prediction in zip(batch, predictions):
                score = prediction["answers"]["automatable"]["noul"]
                results.append(
                    {
                        "rowNumber": ticket["rowNumber"],
                        "score": float(score),
                        "error": None,
                    }
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            for ticket in batch:
                results.append(
                    {
                        "rowNumber": ticket["rowNumber"],
                        "score": None,
                        "error": error,
                    }
                )
            print(f"Batch {batch_number} failed: {error}", file=sys.stderr, flush=True)

        print(
            f"Completed local batch {batch_number} / {total_batches}",
            file=sys.stderr,
            flush=True,
        )

    json.dump(
        {
            "results": results,
            "inference": {
                "startedAt": inference_started_at.isoformat(timespec="seconds"),
                "finishedAt": inference_finished_at.isoformat(timespec="seconds"),
                "seconds": inference_seconds,
                "checkpoints": checkpoints,
            },
        },
        sys.stdout,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    main()