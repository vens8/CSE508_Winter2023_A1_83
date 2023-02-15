import os

path = r"C:\Users\91911\Desktop\CSE508_Winter2023_Dataset\CSE508_Winter2023_Dataset"

os.chdir(path)


def update_text_file(file_path):
    lines = []
    teline = []
    tiline = []
    a = 0
    b = 0
    c = 0
    d = 0
    with open(file_path, 'rt') as myfile:  # Open lorem.txt for reading
        for myline in myfile:  # For each line, read to a string,
            lines.append(myline)
        # print(lines)
    for i in range(len(lines)):
        if lines[i] == "<TITLE>\n":
            a = i
        if lines[i] == "</TITLE>\n":
            b = i
        if lines[i] == "<TEXT>\n":
            c = i
        if lines[i] == "</TEXT>\n":
            d = i
    for i in range(a + 1, b):
        tiline.append(lines[i])
    for i in range(c + 1, d):
        teline.append(lines[i])

    d1 = ""
    for string in tiline:
        d1 = d1 + string[0:len(string) - 1] + " "
    for string in teline:
        d1 = d1 + string[0:len(string) - 1] + " "
    # print(d1)
    with open(file_path,'w+') as myfile:
        myfile.write(d1)
        print(myfile.read())



for file in os.listdir():
    if file.endswith(""):
        file_path = f"{path}\{file}"
        update_text_file(file_path)
