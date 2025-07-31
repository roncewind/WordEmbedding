# -----------------------------------------------------------------------------
# Read wikidata source and create data file of biznames.
# Example usage:
# pbzip2 -d -c -m200 ~/data/BizNamesData/wikidata-20250526-all.json.bz2| python extract_wikidata_pets.py -i - -o data/20250526_pets_wikidata.csv 2> 20250526_err.out
# pbzip2 -d -c -m200 ~/data/BizNamesData/wikidata-20250707-all.json.bz2| python extract_wikidata_pets.py -i - -o data/20250707_pets_wikidata.csv 2> 20250707_err.out
# Output format:
#   wikidata id,canonical name,language code,name
#
# Download Wikidata from: https://dumps.wikimedia.org/wikidatawiki/entities/
# Note: the download is large and takes quite some time,
#   so it's best to download from a dated directory
# EG:
# curl --retry 9 -C - -L -R -O https://dumps.wikimedia.org/wikidatawiki/entities/20250707/wikidata-20250707-all.json.bz2
#

# -----------------------------------------------------------------------------
import argparse
import bz2
import concurrent.futures
import csv
import gzip
import itertools
import sys
import time
import traceback
import unicodedata
from datetime import datetime

import orjson
import script_classifier as script

# -----------------------------------------------------------------------------
# constants
# -----------------------------------------------------------------------------

file_path = ""
debug_on = True
number_of_lines_to_process = 0
number_of_names_to_process = 0
status_print_lines = 10000

# TUNE HERE
INSTANCE_OF_WHITELIST = {
    "Q57814795": "domesticated mammal",
    "Q39201": "pet",  # deprecated in wikidata

}


# =============================================================================
def debug(text):
    if debug_on:
        print(text, file=sys.stderr, flush=True)


# =============================================================================
def format_seconds_to_hhmmss(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


# =============================================================================
# get the canonical name for this entity
def getAsciiName(entity):
    en_name = entity.get("labels", {}).get("en", {}).get("value", "")
    # if the 'en' label doesn't exist look for other forms of an 'en' label
    if len(en_name.strip()) == 0:
        en_name = entity.get("labels", {}).get("en-gb", {}).get("value", "")
        # debug(f'>>en-gb:{entity["id"]}: {en_name}')
    if len(en_name.strip()) == 0:
        en_name = entity.get("labels", {}).get("en-ca", {}).get("value", "")
        # debug(f'>>en-ca:{entity["id"]}: {en_name}')
    # if the label doesn't exist then look for a sitelink title
    if len(en_name.strip()) == 0:
        # debug(f'>>sitelink:{entity["id"]}:{entity.get("sitelinks", {}).get("enwiki", {}).get("title", "")}')
        en_name = entity.get("sitelinks", {}).get("enwiki", {}).get("title", "")
    # if the label and sitelink title don't exist, look for an en alias
    if len(en_name.strip()) == 0:
        # debug(f'>>{entity["id"]}:{entity.get("aliases", {}).get("en", {})[0].get("value", "")}')
        alias_list = entity.get("aliases", {}).get("en", {})
        if alias_list:
            en_name = alias_list[0].get("value", "")
    # bail when we still haven't found an English name to work with
    if len(en_name.strip()) == 0:
        return ""

    # see if there is already an ascii name alias
    if not script.wordIsIn(en_name, script.ascii):
        en_aliases = entity.get("aliases", {}).get("en", {})
        for alias in en_aliases:
            val = alias.get("value")
            if val and script.wordIsIn(val, script.ascii):
                # debug(f'>>return alias:{entity["id"]}: {val}')
                return val.lower()

    # normalize any other non-ascii characters
    en_name = unicodedata.normalize('NFKD', en_name).encode('ascii', 'ignore').decode('ascii')
    # debug(f'>>return:{entity["id"]}: {en_name.lower()}')
    return en_name.lower()


# =============================================================================
# process with black listed properties
def process_json_line_whitelisted(line):
    # Parse the JSON string
    entity = orjson.loads(line)
    out_string = f'{entity["id"]} : {entity["type"]}'
    file_output_dict = {}
    file_should_write = False
    # get the english name
    en_name = getAsciiName(entity)
    if en_name == "":
        debug(f'!! No canonical name for {entity["id"]}')
        return
    # TUNE HERE
    # instance_of = entity.get("claims", {}).get("P31")  # P31 - instance of
    instance_of = entity.get("claims", {}).get("P279")  # P279 - subclass of
    out_string += f' : {en_name}(en)'
    if instance_of:
        for inst in instance_of:
            prop_id = inst.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
            prop_value = INSTANCE_OF_WHITELIST.get(prop_id)
            if prop_value:
                out_string += f' whitelisted[for {prop_value} {prop_id}] '
                file_should_write = True
                break
    else:
        # require that each item be an instance of something.
        # debug(f'no instance of for {entity["id"]}')
        return
    if file_should_write:
        # look for labels to add
        if entity["labels"]:
            for lang in entity["labels"]:
                lang_prop = entity["labels"][lang]
                local_name = lang_prop["value"]
                file_output_dict[local_name] = [entity["id"], en_name, lang]
                #   out_string += f' : {local_name}({lang_prop["language"]})[{rune_count}]'
        # look for aliases to add
        if entity["aliases"]:
            for lang in entity["aliases"]:
                lang_prop = entity["aliases"][lang]
                for prop in lang_prop:
                    local_name = prop["value"]
                    file_output_dict[local_name] = [entity["id"], en_name, lang]
                    #   out_string += f' : {local_name}({prop["language"]})[{rune_count}]'
        debug(out_string)
        return file_output_dict


# =============================================================================
# process a line from the file
def process_line(line):
    stripped_line = line.strip()
    if "" == stripped_line or len(stripped_line) < 10:
        return
    # strip off the comma
    if stripped_line.endswith(","):
        stripped_line = stripped_line[:-1]
    return process_json_line_whitelisted(stripped_line)


# =============================================================================
# Read a jsonl file and process concurrently
def read_file_futures(file_handle, output_file_path):
    line_count = 0
    name_count = 0
    shutdown = False
    start_time = time.time()
    with open(output_file_path, "w") as cjk_out:
        writer = csv.writer(cjk_out)
        writer.writerow(['id', 'canonical', 'language', 'name'])
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_line, line): line
                for line in itertools.islice(file_handle, executor._max_workers * 10)
            }

            while futures:
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for f in done:
                    try:
                        file_output_dict = f.result()
                        if file_output_dict is not None:
                            if id is not None:
                                name_count += 1
                                for key, value in file_output_dict.items():
                                    writer.writerow([*value, key])
                    except Exception:
                        pass
                    else:
                        if not shutdown:
                            line = file_handle.readline()
                            if line:
                                futures[executor.submit(process_line, line)] = (line)
                                line_count += 1
                            if line_count % status_print_lines == 0:
                                print(f"☺ {line_count:,} lines read in {format_seconds_to_hhmmss(time.time() - start_time)}", flush=True, end='\r')
                            if number_of_lines_to_process > 0 and line_count >= number_of_lines_to_process:
                                executor.shutdown(wait=True, cancel_futures=False)
                                shutdown = True
                            if number_of_names_to_process > 0 and name_count >= number_of_names_to_process:
                                executor.shutdown(wait=True, cancel_futures=False)
                                shutdown = True
                    finally:
                        del futures[f]
    print(f"\n{line_count:,} total lines read", flush=True)
    print(f"{name_count:,} total names found", flush=True)
    print(f"{format_seconds_to_hhmmss(time.time() - start_time)} total time", flush=True)
    return line_count


# =============================================================================
# Read a jsonl file and process
def read_file(file_handle, output_file_path):
    line_count = 0
    name_count = 0
    start_time = time.time()
    with open(output_file_path, "w") as cjk_out:
        writer = csv.writer(cjk_out)
        for line in file_handle:
            try:
                file_output_dict = process_line(line)
                if file_output_dict is not None:
                    id = file_output_dict["id"]
                    if id is not None:
                        name_count += 1
                        del file_output_dict["id"]
                        for key, value in file_output_dict.items():
                            writer.writerow([id, key, *value])
            except Exception:
                pass
            if line_count % status_print_lines == 0:
                print(f"{line_count:,} lines read in {format_seconds_to_hhmmss(time.time() - start_time)} seconds", flush=True)
            if number_of_lines_to_process > 0 and line_count >= number_of_lines_to_process:
                break
            if number_of_names_to_process > 0 and name_count >= number_of_names_to_process:
                break
    print(f"{line_count:,} total lines read", flush=True)
    print(f"{name_count:,} total names found", flush=True)
    print(f"{format_seconds_to_hhmmss(time.time() - start_time)} total time", flush=True)
    return line_count


# =============================================================================
# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="lex", description="Creates CSV file business names."
    )

    parser.add_argument("-i", "--infile", action="store", required=True)
    parser.add_argument("-o", "--outfile", action="store", required=True)
    # parser.add_argument("infile", type=str, help="File of lines to phrase")
    args = parser.parse_args()

    infile_path = args.infile
    outfile_path = args.outfile
    line_count = 0
    start_time = datetime.now()
    print("☺", flush=True, end='\r')

    try:
        if infile_path.endswith(".bz2"):
            debug(f"Opening {infile_path}...")
            with bz2.open(infile_path, 'rt') as f:
                line_count = read_file_futures(f, outfile_path)
        elif infile_path.endswith(".gz"):
            debug(f"Opening {infile_path}...")
            with gzip.open(infile_path, 'rt') as f:
                line_count = read_file_futures(f, outfile_path)
        elif infile_path == "-":
            debug("Opening stdin...")
            with open(sys.stdin.fileno(), 'rt') as f:
                line_count = read_file_futures(f, outfile_path)
        else:
            debug("Unrecognized file type.")
    except Exception:
        traceback.print_exc()

    end_time = datetime.now()
    print(f"Input read from {infile_path}, output written to {outfile_path}.", flush=True)
    print(f"    Started at: {start_time}, ended at: {end_time}.", flush=True)
