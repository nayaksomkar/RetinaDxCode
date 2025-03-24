import pandas as pd

# Load the CSV file
#df = pd.read_excel("C:\someFiles\githubRepo\RetinaDxCode\LeftFundusDSB.xlsx")

# Get all values from a column (e.g., 'column_name')
#values = df["Left-Diagnostic Keywords"].tolist()  # Convert to list if needed

filename = "C:/someFiles/githubRepo/RetinaDxCode/RightFundusDSB.xlsx"
df = pd.read_excel(filename)

# Get all values from a column (e.g., 'column_name')
#values += df["Right-Diagnostic Keywords"].tolist()  # Convert to list if needed

def dataExtract(filename, listofColumns):
    df = pd.read_excel(filename)
    
    keywords = df[listofColumns[0]].tolist()
    N = df[listofColumns[1]].tolist()
    D = df[listofColumns[2]].tolist()           
    G = df[listofColumns[3]].tolist()
    C = df[listofColumns[4]].tolist()
    A = df[listofColumns[5]].tolist()
    H = df[listofColumns[6]].tolist()
    M = df[listofColumns[7]].tolist()
    O = df[listofColumns[8]].tolist()

    return {listofColumns[0]: keywords, "N": N, "D": D, "G": G, "C": C, "A": A, "H": H, "M": M, "O": O}

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