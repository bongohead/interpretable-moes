from datasets import load_dataset
import json

# There is no randomness here, we will shuffle the order of generated shards and also use shuffle=True in the DataLoader
def get_fineweb_edu_data_sharded(
    shard_size = 50000,
    max_samples = None,
    out_prefix = "./train_shard",
    val_filename = "./val_shard.json",
    val_size = 1000
):
    """
    Stream the FineWeb-Edu dataset and write out training samples in shards. Also create a validation shard of 'val_size' samples at the beginning.
    These are stored as raw samples.

    Params:
        @shard_size: Number of samples per training shard.
        @max_samples: Total samples for training. If `None`, read until dataset ends.
        @out_prefix: Filename prefix for train shards.
        @val_filename: Filename for the validation shard.
        @val_size: Number of samples in the validation set.
    """
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=True)
    ds = ds.filter(lambda x: x.get("language") == "en") #and x.get("score") >= 4
    ds_iter = iter(ds)

    # ------------------------------------------------
    # Collect validation samples
    # ------------------------------------------------
    val_data = []
    for _ in range(val_size):
        sample = next(ds_iter, None)
        if sample is None:
            break
        val_data.append(sample["text"])

    with open(val_filename, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False)
    print(f"Saved {len(val_data)} validation samples to {val_filename}")

    # ------------------------------------------------
    # Collect training shards in a single pass
    # ------------------------------------------------
    total_written = 0
    shard_idx = 0

    while True:
        # If we have a max_samples limit and we've reached it, stop
        if max_samples is not None and total_written >= max_samples:
            break

        # Gather up to shard_size items
        chunk = []
        for _ in range(shard_size):
            sample = next(ds_iter, None)
            if sample is None:
                # No more data in the stream
                break
            chunk.append(sample)

        if not chunk:
            break  # We reached EOF on the stream

        # Extract text from each sample
        texts = [x["text"] for x in chunk]

        # Write shard
        shard_path = f"{out_prefix}_{shard_idx}.json"
        with open(shard_path, "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)

        shard_idx += 1
        total_written += len(chunk)
        print(f"Wrote shard {shard_path} with {len(chunk)} samples (total so far: {total_written}).")

    print("Done generating shards.")

get_fineweb_edu_data_sharded()