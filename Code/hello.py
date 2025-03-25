import pandas as pd

# Load the CSV file
#df = pd.read_excel("C:\someFiles\githubRepo\RetinaDxCode\LeftFundusDSB.xlsx")

# Get all values from a column (e.g., 'column_name')
#values = df["Left-Diagnostic Keywords"].tolist()  # Convert to list if needed

filename = "C:/someFiles/githubRepo/RetinaDxCode/RightFundusDSB.xlsx"
df = pd.read_excel(filename)

# Get all values from a column (e.g., 'column_name')
#values += df["Right-Diagnostic Keywords"].tolist()  # Convert to list if needed



Leftdata = dataExtract("C:\someFiles\githubRepo\RetinaDxCode\LeftFundusDSB.xlsx", ["Left-Diagnostic Keywords", "N", "D", "G", "C", "A", "H", "M", "O"])
#print(data["Left-Diagnostic Keywords"])
Rightdata = dataExtract("C:\someFiles\githubRepo\RetinaDxCode\RightFundusDSB.xlsx", ["Right-Diagnostic Keywords", "N", "D", "G", "C", "A", "H", "M", "O"])
#print(Rightdata["N"])


# Set all values to 0 first
df[["N", "D", "G", "C", "A", "H", "M", "O"]] = 0

for i in range(len(df)):
    keywords = str(df.at[i, "Right-Diagnostic Keywords"]).lower()


    if 'normal fundus' in keywords:
        df.at[i, "N"] = 1
    else:
        df.at[i, "N"] = 0

    if 'proliferative retinopathy' in keywords:
        df.at[i, "D"] = 1
    else:
        df.at[i, "D"] = 0

    if 'glaucoma' in keywords:
        df.at[i, "G"] = 1
    else:
        df.at[i, "G"] = 0

    if 'cataract' in keywords:
        df.at[i, "C"] = 1
    else:
        df.at[i, "C"] = 0

    if 'age-related macular degeneration' in keywords:
        df.at[i, "A"] = 1
    else:
        df.at[i, "A"] = 0

    if 'hypertensive retinopathy' in keywords:
        df.at[i, "H"] = 1
    else:
        df.at[i, "H"] = 0

    if 'myopia' in keywords:
        df.at[i, "M"] = 1
    else:
        df.at[i, "M"] = 0

    # If none of the conditions above are met, set "O" to 1
    if not (df.at[i, "N"] or df.at[i, "D"] or df.at[i, "G"] or df.at[i, "C"] or df.at[i, "A"] or df.at[i, "H"] or df.at[i, "M"]):
        df.at[i, "O"] = 1
    else:
        df.at[i, "O"] = 0



print(df)
df.to_excel(filename, index=False)

















import pandas as pd

filepath = r"C:\Users\nayak\Downloads\Dataset B\data.xlsx"
# Load CSV filed
df = pd.read_excel(filepath) 
 # Display first 5 rows
print(df.head())

# Select the columns you want to write to the new file
selected_columns = ["ID", "Left-Fundus", "Left-Diagnostic Keywords", "N", "D", "G", "C", "A", "H", "M", "O"] 
# Create a new DataFrame with selected columns
new_df = df[selected_columns] 
# Save to a new Excel file
new_df.to_excel("LeftFundusDSB.xlsx", index=False, engine="openpyxl")
print("New file created successfully!")

selected_columns = ["ID", "Right-Fundus", "Right-Diagnostic Keywords", "N", "D", "G", "C", "A", "H", "M", "O"]  
new_df = df[selected_columns]
# Save to a new Excel file
new_df.to_excel("RightFundusDSB.xlsx", index=False, engine="openpyxl")

print("New file created successfully!")