import pandas as pd
import os 
import os
import shutil

#create folder for each image category
folderpath = r"C:\Users\nayak\Downloads\Dataset X"
os.makedirs(folderpath, exist_ok=True)
#create folder name normal
os.makedirs(os.path.join(folderpath, "normal"), exist_ok=True)
os.makedirs(os.path.join(folderpath, "diabetic_retinopathy"), exist_ok=True)
os.makedirs(os.path.join(folderpath, "glaucoma"), exist_ok=True)    
os.makedirs(os.path.join(folderpath, "cataract"), exist_ok=True)
os.makedirs(os.path.join(folderpath, "age-related macular degeneration"), exist_ok=True)
os.makedirs(os.path.join(folderpath, "hypertensive retinopathy"), exist_ok=True)
os.makedirs(os.path.join(folderpath, "myopia"), exist_ok=True)
os.makedirs(os.path.join(folderpath, "other"), exist_ok=True)

def copyImage(foldername, imagePath):
    destination = os.path.join(folderpath, foldername)
    os.makedirs(destination, exist_ok=True)
    shutil.copy(imagePath, os.path.join(destination, os.path.basename(imagePath)))

for prefix in ["Left", "Right"]:
    filepath  = f"C:\someFiles\githubRepo\RetinaDxCode\{prefix}FundusDSB.xlsx"
    #folderpath = f"C:\Users\nayak\Downloads\Dataset X"

    # Load CSV filed
    df = pd.read_excel(filepath)
    selected_columns = [f"{prefix}-Fundus", "N", "D", "G", "C", "A", "H", "M", "O"]
    df = df[selected_columns]
    #print(df.head())

    for index in range(len(df)):
        imageName = df.at[index, f"{prefix}-Fundus"]
        imagePath = rf"C:\Users\nayak\Downloads\Dataset B\Images\{imageName}"


        if df.at[index, "N"] == 1:
            copyImage(foldername = 'normal', imagePath = imagePath)

        if df.at[index, "D"] == 1:
            copyImage(foldername = 'diabetic_retinopathy', imagePath = imagePath)

        if df.at[index, "G"] == 1:
            copyImage(foldername = 'glaucoma', imagePath = imagePath)

        if df.at[index, "C"] == 1:
            copyImage(foldername = 'cataract', imagePath = imagePath)

        if df.at[index, "A"] == 1:
            copyImage(foldername = 'age-related macular degeneration', imagePath = imagePath)

        if df.at[index, "H"] == 1:
            copyImage(foldername = 'hypertensive retinopathy', imagePath = imagePath)

        if df.at[index, "M"] == 1:
            copyImage(foldername = 'myopia', imagePath = imagePath)

        if df.at[index, "O"] == 1:
            copyImage(foldername = 'other', imagePath = imagePath)
            
    print("Images copied to respective folders successfully!")
