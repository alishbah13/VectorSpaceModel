import tkinter as tk
from tkinter import ttk
from tkinter import *
from vsm import *

def show():

    user_query = raw_query.get()

    x = vector_space_model()
    x.get_files('ShortStories')
    x.get_raw_text('ShortStories')
    x.preprocess()
    x.set_index() 
    x.store_index()
    x.tf_idf(user_query)

    result_list = x.cosine_sim()
    print(result_list)
    

    for i, ( name) in enumerate(result_list, start=1):
        listBox.insert("", "end", values=( i,name[0], name[1]))
    listBox.insert("", "end", values=( "-", "-", "-"))

root = tk.Tk()
root.title('Vector Space Model')
windowWidth = root.winfo_reqwidth()
windowHeight = root.winfo_reqheight()
positionRight = int(root.winfo_screenwidth()/2 - windowWidth)
positionDown = int(root.winfo_screenheight()/2 - windowHeight)
root.geometry("+{}+{}".format(positionRight, positionDown)) 


raw_query = StringVar()

label = tk.Label(root, text="Enter query ").grid(row=0,column =0, columnspan=3)
entry = tk.Entry(root, text="",textvariable = raw_query).grid(row=1, column=0, columnspan=3)
button = tk.Button(root, text="Search",command=show).grid(row=2, column=0, columnspan=3)



# create Treeview with 2 columns
cols = ('Rank', 'Document ID', 'Document Name')
listBox = ttk.Treeview(root, columns=cols, show='headings', height = 20)

# set column headings
for col in cols:
    listBox.heading(col, text=col)    
listBox.grid(row=3, column=0, columnspan=2)

closeButton = tk.Button(root, text="Close", width=15, command=exit).grid(row=4, columnspan=3)

root.mainloop()