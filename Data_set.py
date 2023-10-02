#           **************************************************************************
#        ///////////////////****SPLIT LINES THAT HAVE 'O' TAGS****\\\\\\\\\\\\\\\\\\\\\\\
#           **************************************************************************

import csv


def process_csv(input_csv_path, output_csv_path):
    with open(input_csv_path, 'r', newline='', encoding='utf-8') as input_csvfile, \
            open(output_csv_path, 'w', newline='', encoding='utf-8') as output_csvfile:

        reader = csv.reader(input_csvfile)
        writer = csv.writer(output_csvfile)

        writer.writerow(["File Name", "Processed Text", "Tags"])  # Write the header row

        for row in reader:
            file_name = row[0]
            processed_text = row[1]
            tags = row[2]

            if "O" in tags:
                processed_text_parts = processed_text.split()
                for text_part in processed_text_parts:
                    writer.writerow([file_name, text_part, tags])
            else:
                writer.writerow(row)


if __name__ == "__main__":
    input_csv_path = "dataset_bccatering.csv"  # Replace with your input CSV file path
    output_csv_path = "output.csv"  # Replace with the desired output CSV file path

    process_csv(input_csv_path, output_csv_path)
    print("CSV file processed and updated")


#        ***********************************************************************************************
#        \\\\\//////////****SPLIT LINES THAT HAVE TAGS B-BESKRIVELSE AND I-BESKRIVELSE****\\\\\\\\//////
#        //////AND ALSO GIVE THEM TAGS TO TOKENS 1ST TOKEN GET 'B' AND FOR ALL OTHER TOKENS GET 'I'\\\\\
#        ************************************************************************************************

#
# import csv
#
#
# def process_csv(input_csv_path, output_csv_path):
#     with open(input_csv_path, 'r', newline='', encoding='utf-8') as input_csvfile, \
#             open(output_csv_path, 'w', newline='', encoding='utf-8') as output_csvfile:
#
#         reader = csv.reader(input_csvfile)
#         writer = csv.writer(output_csvfile)
#
#         writer.writerow(["File Name", "Processed Text", "Tags"])  # Write the header row
#
#         for row in reader:
#             file_name = row[0]
#             processed_text = row[1]
#             tags = row[2]
#
#             if "B-BESKRIVELSE" in tags or "I-BESKRIVELSE" in tags:
#                 processed_text_parts = processed_text.split()
#                 for i, text_part in enumerate(processed_text_parts):
#                     if i == 0:
#                         new_tags = "B-BESKRIVELSE"
#                     else:
#                         new_tags = "I-BESKRIVELSE"
#                     writer.writerow([file_name, text_part, new_tags])
#             else:
#                 writer.writerow(row)
#
#
# if __name__ == "__main__":
#     input_csv_path = "output.csv"  # Replace with your input CSV file path
#     output_csv_path = "output4.csv"  # Replace with the desired output CSV file path
#
#     process_csv(input_csv_path, output_csv_path)
#     print("CSV file processed and updated")
#
