def add_dots(num_str):
    return "{:,}".format(int(num_str)).replace(",", ".")

def limit_decimal_points(num_str, n_decimal=2):
    # Convert the string to a float and format it with the specified number of decimal places
    return f"{float(num_str):.{n_decimal}f}"

def get_struct_height(values, index):
    height = 1  # Start bei 1, da der Wert mindestens einmal vorhanden ist
    current_value = values[index]

    # Überprüfen Sie die folgenden Elemente, um Wiederholungen zu zählen
    for i in range(index + 1, len(values)):
        if values[i] == current_value:
            height += 1
        else:
            break  # Beenden, wenn der nächste Wert anders ist

    return height

def create_tabular(rows_idx=["l", "l", "r", "r", "r", "r"], rows=[], data={}):
   for i in range(len(data[rows[0]["name"]])):
      line = ""
      for idx_row, row in enumerate(rows):
        value = str(data[row["name"]][i])
        
        # Add Dots for 1000
        if row["type"] == "integer":
           value = add_dots(value)
           
        # Limit Decimal Points
        if row["type"] == "double":
           if "decimal_place" in row.keys():  # Check if 'decimal_place' key exists
               if not("-" in value):
                  value = limit_decimal_points(value, row["decimal_place"])
               
        # Structure row
        
        if row["struct"] == True:
           all_values_in_row = [data[row["name"]][i] for i in range(len(data[row["name"]]))]  # Get all values in the row
           if (data[row["name"]][i] == data[row["name"]][i-1]) and i != 0: ### !!! Evtl wieder ändern
              value = ""
           elif len(set(all_values_in_row)) > 1 or i == 0:
              value = "\\multirow{"+str(get_struct_height(data[row["name"]], i))+"}{*}{"+ value + "}"
                       
        # Add bold text
        if row["struct"] == True:
           value = "\\textbf{"+value+"}"
           
          
        line += value + " & "
        
          
        # Add Big Line
        try:
           print_hline = False
           if (idx_row) == 0 and (data[row["name"]][i] != data[row["name"]][i-1]) and (data[row["name"]][i] == data[row["name"]][i+1]):
             print("\\hline")
             print_hline = True
             
           if (row["struct"] == True) & (idx_row != 0) & (print_hline == False):
              try:
                 if (data[row["name"]][i] != data[row["name"]][i-1]):
                    print("\\arrayrulecolor{gray}\\cline{" + str(idx_row+1) + "-" + str(len(rows)) + "}\\arrayrulecolor{black}")
                    
              except:
                 pass                 
        except:
           pass
              
      print(line[:-2] + "\\\\")