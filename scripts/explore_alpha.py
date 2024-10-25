import os
import sys
import time


def create_bash_script(confidence_file, templates_folder):
    """
    Creates a bash script that performs the LSH operations.

    Args:
      confidence_file: Path to the confidence file.
      templates_folder: Path to the templates folder.
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_folder = os.path.join("subsets", timestamp)

    os.makedirs(output_folder, exist_ok=True)

    confidence_file_name = os.path.basename(confidence_file)

    templates_folder_name = os.path.basename(templates_folder)

    with open("run_lsh.sh", "w") as f:
        f.write("#!/bin/bash\n\n")

        for alpha in [0, 0.5, 1, 2, 3, 5, 8, 10, 15, 20]:
            for size in [60, 65, 70, 75, 80, 85, 90, 95, 100]:
                output_file = f"subset_alpha_{alpha}_size_{size}_{confidence_file_name}_count_250000.bin"
                f.write(
                    f"./target/release/lsh-lock zeta-sampling --alpha {alpha} --confidence {confidence_file} --count 250000 --output {output_folder}/{output_file} --size {size}\n"
                )

        output_file_analyze = f"{output_folder}/{confidence_file_name}_{templates_folder_name}_entropy.output"
        f.write(f"> {output_file_analyze}\n")
        for alpha in [0, 0.5, 1, 2, 3, 5, 8, 10, 15, 20]:
            for size in [60, 65, 70, 75, 80, 85, 90, 95, 100]:
                input_file = f"{output_folder}/subset_alpha_{alpha}_size_{size}_{confidence_file_name}_count_250000.bin"
                f.write(
                    f"./target/release/lsh-lock analyze --count 10 --input '{input_file}' --templates {templates_folder} >> {output_file_analyze}\n"
                )

        output_file_tar = (
            f"{output_folder}/{confidence_file_name}_{templates_folder_name}_tar.output"
        )
        f.write(f"> {output_file_tar}\n")
        for alpha in [0, 0.5, 1, 2, 3, 5, 8, 10, 15, 20]:
            for size in [60, 65, 70, 75, 80, 85, 90, 95, 100]:
                input_file = f"{output_folder}/subset_alpha_{alpha}_size_{size}_{confidence_file_name}_count_250000.bin"
                f.write(
                    f"./target/release/lsh-lock tar --count 250000 --input '{input_file}' --templates {templates_folder} >> {output_file_tar}\n"
                )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python explore_alpha.py <confidence_file> <templates_folder>")
        sys.exit(1)

    confidence_file = sys.argv[1]
    templates_folder = sys.argv[2]

    create_bash_script(confidence_file, templates_folder)
    print("Bash script 'run_lsh.sh' created successfully.")
