def bolden_table_max_values(printed_output, ignore_columns):
    tabular_split = printed_output.split("\\\\")
    columns = []

    for i, line in enumerate(tabular_split):
        line_items = line.split(" & ")
        line_items = [item.strip() for item in line_items if item.strip() != ""]
        if len(line_items) > 0:
            columns.append(line_items)

    def find_max_indices(values):
        """
        Finds the indices of the maximum value in a list of floats.

        :param values: List of float numbers
        :return: List of indices where the maximum value occurs
        """
        if not values:  # Handle empty list
            return []

        max_value = max(values)  # Find the maximum value
        return [i for i, v in enumerate(values) if v == max_value]  # Find all indices of max_value

    for i in range(len(columns[0])):
        if i in ignore_columns:
            continue

        values_to_compare = [float(columns[k][i]) for k in range(len(columns)) if columns[k][i] != "-"]

        # bolden the max value in the column
        for index in find_max_indices(values_to_compare):
            columns[index][i] = "\\textbf{" + columns[index][i] + "}"

    # merge the columns back together
    for i, line in enumerate(columns):
        columns[i] = " & ".join(line)

    printed_output = " \\\\ \n ".join(columns) + " \\\\ \n"

    return printed_output

def bolden_table_max_values_with_hline(printed_output, ignore_columns):
    out_txt = ""
    printed_output = [output for output in printed_output.split("\n\\hline\n") if output.strip() != ""]
    for output in printed_output:
       out_txt += bolden_table_max_values(output, ignore_columns)
       out_txt += "\hline\n"
    return out_txt