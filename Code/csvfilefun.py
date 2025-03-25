import pandas as pd

def fixcsvvalues(filepath):
    # Load CSV filed
    masterdf = pd.read_excel(filepath) 
    # Set all values to 0 (False)
    masterdf[["N", "D", "G", "C", "A", "H", "M", "O"]] = 0
    # Display first 5 rows
    #print(df.head())   

    for prefix in ["Left", "Right"]:
        selected_columns = ["ID", f"{prefix}-Fundus", f"{prefix}-Diagnostic Keywords", "N", "D", "G", "C", "A", "H", "M", "O"] 
        df =  masterdf[selected_columns] 
        # Save to a new Excel file
        df.to_excel(f"{prefix}FundusDSB.xlsx", index=False, engine="openpyxl")
        print("New file created successfully!")
        
        for i in range(len(df)):
            keywords = str(df.at[i, f"{prefix}-Diagnostic Keywords"]).lower()

            Other = 0 #This is a flag to check if any of the keywords are present
            if 'normal fundus' in keywords:
                df.at[i, "N"] = Other = 1 
            else:
                df.at[i, "N"] = 0

            if 'proliferative' in keywords:
                df.at[i, "D"] = Other = 1
            else:
                df.at[i, "D"] = 0

            if 'glaucoma' in keywords:
                df.at[i, "G"] = Other = 1
            else:
                df.at[i, "G"] = 0

            if 'cataract' in keywords:
                df.at[i, "C"] = Other = 1
            else:
                df.at[i, "C"] = 0

            if 'age-related macular degeneration' in keywords:
                df.at[i, "A"] = Other = 1
            else:
                df.at[i, "A"] = 0

            if 'hypertensive retinopathy' in keywords:
                df.at[i, "H"] = Other = 1
            else:
                df.at[i, "H"] = 0

            if 'myopia' in keywords:
                df.at[i, "M"] = Other = 1
            else:
                df.at[i, "M"] = 0

            if Other:
                df.at[i, "O"] = 0
            else:
                df.at[i, "O"] = 1

        df.to_excel(f"{prefix}FundusDSB.xlsx", index=False, engine="openpyxl")
        print("Truth Values updated based on keywords!")

    

filepath = r"C:\Users\nayak\Downloads\Dataset B\data.xlsx"
fixcsvvalues(filepath=filepath)