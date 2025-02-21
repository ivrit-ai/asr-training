import json
from random import shuffle
import logging
from pathlib import Path
from typing import Iterator
import uuid

from tqdm import tqdm
from stable_whisper.result import WhisperResult, Segment
from audiosample import AudioSample
from datasets import Dataset, DatasetDict, concatenate_datasets


logger = logging.getLogger(__name__)


def _load_data_manifest(
    input_folder: Path,
    audio_filename_glob: str,
    segments_glob: str,
    metadata_glob: str,
):
    segments_files = list(input_folder.glob(segments_glob))
    audio_files = []
    metadata_files = []
    for segments_file in segments_files:
        # find the audio file that has matches the glob and within the same directory
        search_within_folder = segments_file.parent
        found_audio_files = list(search_within_folder.glob(audio_filename_glob))
        # expect only one audio file
        assert (
            len(found_audio_files) == 1
        ), f"Expected 1 audio file, found {len(found_audio_files)} for {segments_file} (taking first)"
        audio_files.extend(found_audio_files[:1])
        # expect only one metadata file
        found_metadata_files = list(search_within_folder.glob(metadata_glob))
        assert (
            len(found_metadata_files) == 1
        ), f"Expected 1 metadata file, found {len(found_metadata_files)} for {segments_file} (taking first)"
        metadata_files.extend(found_metadata_files[:1])
    return list(zip(audio_files, segments_files, metadata_files))


from subprocess import CalledProcessError, run, PIPE, DEVNULL

import numpy as np

WHISPER_EXPECTED_SAMPLE_RATE = 16000


def load_audio_in_whisper_format(file: str, sr: int = WHISPER_EXPECTED_SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary
    Shamelessly stolen from https://github.com/openai/whisper/blob/main/whisper/audio.py
    Thanks OpenAI :)

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def create_captions_from_segments(segments_data: WhisperResult):
    captions = []
    for segment in segments_data.segments:
        captions.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }
        )
    return captions


def calculate_segment_quality_score(segment: Segment) -> float:
    """Calculate the quality score based on the median word probabilities for a single segment."""
    try:
        word_probs = []
        if segment.has_words:
            for word in segment.words:
                if hasattr(word, "probability"):
                    word_probs.append(word.probability)

        quality_score = float(np.median(word_probs)) if word_probs else 0.0
        return quality_score

    except Exception:
        return 0.0


def generate_slices(
    input_segments: list[Segment],
    audio_duration: float,
    slice_length: int,
    per_segment_quality_threshold: float = 0,
):
    next_slice_start = 0
    curr_input_segment_idx = 0
    slices = []
    while next_slice_start < audio_duration:
        slice_start = next_slice_start

        curr_slice_segments = []
        curr_slice = {"segments": curr_slice_segments, "seek": slice_start}
        slices.append(curr_slice)
        # normal slice length is the expected slice hop - but this could be overridden below. See comments.
        next_slice_start = slice_start + slice_length
        # clip the slice end to the audio duration
        slice_end = min(next_slice_start, audio_duration)
        while curr_input_segment_idx < len(input_segments) and input_segments[curr_input_segment_idx].start < slice_end:
            curr_input_segment = input_segments[curr_input_segment_idx]
            segment_quality_score = calculate_segment_quality_score(curr_input_segment)

            # Low quality segments are ignored
            # We assume they represent text which is not in the audio.
            if segment_quality_score < per_segment_quality_threshold:
                low_quality_segment_idx = curr_input_segment_idx
                curr_input_segment_idx += 1
                # If the low quality segment is not the first or the last in the entire
                # audio sample, we will discard the entire slice in case it contains
                # garbage
                if low_quality_segment_idx > 0 and low_quality_segment_idx < len(input_segments) - 1:
                    curr_slice_segments.clear()
                    next_slice_start = input_segments[low_quality_segment_idx].end
                    break
                else:
                    # in the edge-low-quality cases, continue the normal slicing flow
                    # as if this segment did not exist
                    continue

            slice_segment = {
                "start": curr_input_segment.start - slice_start,  # relative to slice
            }
            curr_slice_segments.append(slice_segment)

            # Clip the segment end to the entire audio duration
            # This is meant to prevent small segment timing overflows over audio
            # duration which stems from arbitrary rounding errors in the data prep
            # and subtitles alignment logic.
            curr_input_segment_end = min(curr_input_segment.end, audio_duration)

            # If this input segment ends within the slice
            # It would be entirely contained including it's text and timestamps
            if curr_input_segment_end <= slice_end:
                #   s   e   s         e
                #  /    \  /          \??????
                # |_________________________|
                #                     ^
                slice_segment["end"] = curr_input_segment_end - slice_start  # relative to slice
                slice_segment["text"] = curr_input_segment.text

                # entire segment is included - no need to reference it again on the next slice.
                curr_input_segment_idx += 1

            # Else - we cannot complete this segment on this slice.
            # The "start" of the segment is kept in the slice to mark it's crossing onto the next
            # slice but the next slice will also need to start at the **end** of the previous segment
            # to allow proper "restart" of the overflowing segment
            else:
                # This slice ends - close this slice

                # Special case - If the "start only" segment is the only one - don't include it at all.
                # Instead, this slice would be left empty.
                if len(curr_slice_segments) == 1:
                    #           s                    e
                    #          /                     \
                    # |_________________________||........
                    #                                ^
                    curr_slice_segments.clear()
                    # In this special case, the current segment starts within
                    # the slice and ends outside of it. But it is the only segment.
                    # We need to start the next slice on the **start** of this segment
                    # and not at the end of the previous one (which is not within this slice
                    # at all
                    next_slice_start = input_segments[curr_input_segment_idx].start
                else:
                    #   s    e  s                    e
                    #  /     \ /                     \
                    # |_________________________||........
                    #                                ^
                    # This is the normal cross-over case.
                    # The current segment starts within this slice
                    # and ends outside of it and other segments within this slice were closed normally.
                    # We need to start the next slice on the **end** of prev segment before the "start-only" one.
                    next_slice_start = input_segments[curr_input_segment_idx - 1].end

                # Break, this slice is done.
                break

    return slices


def get_slice_audio_data(audio_data, slice, slice_length):
    audio_start_sec = slice["seek"]
    audio_end_sec = audio_start_sec + slice_length
    return audio_data[audio_start_sec:audio_end_sec]


def get_timestamp_token_text(seconds: float) -> str:
    """
    Get the timestamp token text for a given seconds.
    This is a helper function to encode the timestamp tokens for the Whisper model.
    It is specific to Whisper and should be moved to a proper util that handles
    timestamp tokens encoding/decoding for any ASR model.
    """
    if 0 <= seconds <= 30:
        # round to precision of .02
        rounded = 0.02 * round(seconds / 0.02)
        return f"<|{rounded:.2f}|>"
    else:
        raise ValueError("Timestamp token out of range.")


def generate_examples_from_slices(slices, slice_length, audio_data, metadata: dict) -> Iterator[dict]:
    source_id = metadata.get("source_id", "unknown")
    source_entry_id = metadata.get("source_entry_id", str(uuid.uuid4()))
    logger.debug(f"Generating dataset from {source_id}/{source_entry_id}")

    # No slices - nothing to do
    if not slices:
        return None

    # At least one segments we can work on is expected
    if next(iter([seg for s in slices for seg in s["segments"]]), None) is None:
        return None

    prev_example = None
    for slice in slices:
        if slice["segments"]:
            try:
                slice_text = ""
                for segment in slice["segments"]:
                    slice_text += get_timestamp_token_text(segment["start"])
                    if "text" in segment:
                        slice_text += f'{segment["text"]}{get_timestamp_token_text(segment["end"])}'
                slice_audio_data = get_slice_audio_data(audio_data, slice, slice_length)
                example = {
                    "audio": {
                        "bytes": slice_audio_data.as_data(no_encode=False, force_out_format="mp3"),
                        "path": source_entry_id,
                    },
                    "transcript": slice_text,
                    "metadata": {
                        "seek": float(slice["seek"]),
                        "source": source_id,
                        "entry_id": source_entry_id,
                    },
                    "has_prev": False,
                    "has_timestamps": True,
                    "prev_transcript": "",
                }
                if prev_example:
                    example["prev_transcript"] = prev_example["transcript"]
                    example["has_prev"] = True
                yield example
                prev_example = example
            except Exception as e:
                logger.error(
                    f'Error processing slice seek {float(slice["seek"]):.2f} in {source_id}:{source_entry_id}: {e}'
                )
        else:
            prev_example = None


def prepare_training_dataset(
    input_folder: Path,
    slice_length: int = 30,
    max_source_entries: int = None,
    audio_filename_glob: str = "audio.*",
    segments_filename_glob: str = "transcript.*.json",
    metadata_glob: str = "metadata.json",
    num_proc: int = 1,
    per_proc_per_chunk_size: int = 10,
    per_sample_quality_threshold: float = 0,
    per_segment_quality_threshold: float = 0,
) -> Dataset:
    """
    Prepare captioned datasets from the input folder.
    Produce audio slices and corresponding text including previous text when available
    Returns a HuggingFace Dataset. Splitting (if needed) should be applied outside this function.
    """
    input_folder = Path(input_folder)
    input_manifest = _load_data_manifest(
        input_folder,
        segments_glob=f"**/{segments_filename_glob}",
        audio_filename_glob=audio_filename_glob,
        metadata_glob=metadata_glob,
    )

    # Shuffle source entries to approach more homogenous shard sizes
    shuffle(input_manifest)

    # Aim for 10 entries per worker within each chunk
    manifest_processing_chunk_size = num_proc * per_proc_per_chunk_size

    def examples_from_entry_generator(input_manifest_shards):
        processed_source_entries = 0
        for audio_file, segments_data_file, metadata_file in input_manifest_shards:
            try:
                # Load captions
                segments_data = WhisperResult(str(segments_data_file))
                segments = segments_data.segments

                # Load metadata
                sample_quality_score = None
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    sample_quality_score = metadata.get("quality_score", None)

                if (
                    sample_quality_score is not None
                    and per_sample_quality_threshold > 0
                    and sample_quality_score < per_sample_quality_threshold
                ):
                    logger.debug(
                        f"Skipping sample {audio_file} with quality score {sample_quality_score} (threshold: {per_sample_quality_threshold})"
                    )
                    continue

                # Load Audio
                with AudioSample(
                    load_audio_in_whisper_format(audio_file, sr=WHISPER_EXPECTED_SAMPLE_RATE),
                    force_read_sample_rate=WHISPER_EXPECTED_SAMPLE_RATE,
                ) as audio_data:
                    audio_duration = audio_data.duration

                    # Create slices of the captions with the intended slice
                    slices = generate_slices(segments, audio_duration, slice_length, per_segment_quality_threshold)

                    # Generate the dataset
                    for example in generate_examples_from_slices(
                        slices,
                        slice_length,
                        audio_data,
                        metadata,
                    ):
                        yield example
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")

            processed_source_entries += 1
            if max_source_entries is not None and processed_source_entries >= max_source_entries:
                break

    input_manifest_chunks = [
        input_manifest[i : i + manifest_processing_chunk_size]
        for i in range(0, len(input_manifest), manifest_processing_chunk_size)
    ]

    # Why? Dataset.from_generator does not properly release memory from the generator
    # after completion. To avoid OOM, we:
    # 1. Generate multiple smaller datasets in chunks
    # 2. Let each chunk's generator get GC'd after completion
    # 3. Concatenate the memory-mapped datasets at the end
    # This trades off some disk I/O for better memory usage, while still
    # maintaining parallel generation within each chunk.
    all_datasets = []
    for input_manifest_chunk in tqdm(input_manifest_chunks, desc="Generating input manifest chunks"):
        all_datasets.append(
            Dataset.from_generator(
                examples_from_entry_generator,
                num_proc=num_proc,
                gen_kwargs={"input_manifest_shards": list(input_manifest_chunk)},
            )
        )

    examples_dataset = concatenate_datasets(all_datasets)

    return examples_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLI to prepare a training dataset from the input folder")
    parser.add_argument(
        "input_folder",
        help="Path to the folder containing audio, transcript, and metadata data in the normalized structure",
    )
    parser.add_argument(
        "--max_source_entries", type=int, default=None, help="Maximum number of source entries to process"
    )
    parser.add_argument("--audio_filename_glob", default="audio.*", help="Glob pattern for audio files")
    parser.add_argument("--segments_filename_glob", default="transcript.*.json", help="Glob pattern for segments files")
    parser.add_argument(
        "--validation_split_size", type=float, default=0, help="Split size for evaluation (between 0 and 1)"
    )
    parser.add_argument("--num_proc", type=int, default=1, help="Number of processes to use")
    parser.add_argument(
        "--per_proc_per_chunk_size",
        type=int,
        default=10,
        help=(
            "Number of entries per process per chunk. "
            "This is a memory usage consideration. This number times the number of processes will define the "
            "amount of memory kept around during the generation of a sub-dataset. "
            "If each sample is large (minutes of audio), this number should be decreased. "
            "If each sample is small (seconds of audio), this number can be increased to increase parallelism efficiency. "
        ),
    )
    parser.add_argument(
        "--per_sample_quality_threshold",
        type=float,
        default=0,
        help="Quality threshold for per-sample quality filtering (0-1 below this threshold the entire sample is dropped)",
    )
    parser.add_argument(
        "--per_segment_quality_threshold",
        type=float,
        default=0,
        help="Quality threshold for per-segment quality filtering (0-1 below this threshold a segment and it's surrounding slice are dropped)",
    )
    parser.add_argument("--push_to_hub", type=bool, default=False, help="Push the dataset to the hub")
    parser.add_argument(
        "--output_dataset_name", type=str, help="Name of the dataset, Omit to not store any dataset (dry-run)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset to set in dataset info",
    )
    parser.add_argument(
        "--dataset_version",
        type=str,
        help="Version of the dataset to set in dataset info",
    )
    parser.add_argument("--push_as_public", type=bool, default=True, help="Push the dataset as public")
    parser.add_argument(
        "--clear_output_dataset_cache_files",
        action="store_true",
        help="Clear the HF cache for the output dataset on disk",
    )

    args = parser.parse_args()

    # Prepare the dataset
    output_dataset = prepare_training_dataset(
        input_folder=args.input_folder,
        max_source_entries=args.max_source_entries,
        audio_filename_glob=args.audio_filename_glob,
        segments_filename_glob=args.segments_filename_glob,
        num_proc=args.num_proc,
        per_proc_per_chunk_size=args.per_proc_per_chunk_size,
        per_sample_quality_threshold=args.per_sample_quality_threshold,
        per_segment_quality_threshold=args.per_segment_quality_threshold,
    )

    if output_dataset:
        if args.dataset_name:
            output_dataset.info.dataset_name = args.dataset_name
        if args.dataset_version:
            output_dataset.info.version = args.dataset_version

        if args.validation_split_size > 0:
            # If a validation split is requested, split the dataset in main
            assert args.validation_split_size < 1.0, "validation_split_size must be a float between 0 and 1"
            temp = output_dataset.train_test_split(test_size=args.validation_split_size)
            output_dataset = DatasetDict({"train": temp["train"], "eval": temp["test"]})

        if args.output_dataset_name:
            if args.push_to_hub:
                if not args.push_as_public:
                    logger.warning("Pushing the dataset to the hub as private")
                output_dataset.push_to_hub(args.output_dataset_name, private=not args.push_as_public)
            else:
                output_dataset.save_to_disk(args.output_dataset_name)

        else:
            # report the created dataset sizes per split
            if isinstance(output_dataset, DatasetDict):
                for split, ds in output_dataset.items():
                    logger.info(f"{split}: {ds.num_rows} samples")
            else:
                logger.info(f"Dataset created with {output_dataset.num_rows} samples")

        if args.clear_output_dataset_cache_files and output_dataset:
            logger.info("Clearing output dataset cache files")
            output_dataset.cleanup_cache_files()
    else:
        logger.warning("No dataset was created")
