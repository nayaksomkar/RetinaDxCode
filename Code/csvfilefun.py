import pandas as pd

def fixcsvvalues(filepath):
    # Load CSV filed
    df = pd.read_excel(filepath) 
    # Display first 5 rows
    #print(df.head())   

    for prefix in ["Left", "Right"]:
        selected_columns = ["ID", f"{prefix}-Fundus", f"{prefix}-Diagnostic Keywords", "N", "D", "G", "C", "A", "H", "M", "O"] 
        df[["N", "D", "G", "C", "A", "H", "M", "O"]] = 0
        new_df = df[selected_columns] 
        # Save to a new Excel file
        new_df.to_excel(f"{prefix}FundusDSB.xlsx", index=False, engine="openpyxl")
        print("New file created successfully!")
        
        for i in range(len(df)):
            keywords = str(df.at[i, f"{prefix}-Diagnostic Keywords"]).lower()
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

            if 'other' in keywords:
                df.at[i, "O"] = 1
            else:
                df.at[i, "O"] = 0

        df.to_excel(f"{prefix}FundusDSB.xlsx", index=False, engine="openpyxl")
        print("File updated successfully!")

filepath = r"C:\Users\nayak\Downloads\Dataset B\data.xlsx"
fixcsvvalues(filepath=filepath)