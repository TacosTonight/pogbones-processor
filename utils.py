def separate_files(input_file, in_file, out_file):
    with open(input_file, "r") as f_in:
        with open(in_file, "w") as f_in_out:
            with open(out_file, "w") as f_out:
                for line in f_in:
                    if line.startswith("IN:"):
                        f_in_out.write(line)
                    elif line.startswith("OUT:"):
                        f_out.write(line)
