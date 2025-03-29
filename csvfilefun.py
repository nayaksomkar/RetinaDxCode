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
        
        for index in range(len(df)):
            keywords = str(df.at[index, f"{prefix}-Diagnostic Keywords"]).lower()

            Other = 0 #This is a flag to check if any of the keywords are present
            if 'normal fundus' in keywords:
                df.at[index, "N"] = Other = 1 

            if 'proliferative' in keywords:
                df.at[index, "D"] = Other = 1

            if 'glaucoma' in keywords:
                df.at[index, "G"] = Other = 1
        
            if 'cataract' in keywords:
                df.at[index, "C"] = Other = 1

            if 'age-related macular degeneration' in keywords:
                df.at[index, "A"] = Other = 1

            if 'hypertensive retinopathy' in keywords:
                df.at[index, "H"] = Other = 1

            if 'myopia' in keywords:
                df.at[index, "M"] = Other = 1

            # If none of the above keywords are present, set Other to 1 ()
            if Other == 0:
                df.at[index, "O"] = 1

        df.to_excel(f"{prefix}FundusDSB.xlsx", index=False, engine="openpyxl")
        print("Truth Values updated based on keywords!")

    

filepath = r"C:\Users\nayak\Downloads\Dataset B\data.xlsx"
fixcsvvalues(filepath=filepath)