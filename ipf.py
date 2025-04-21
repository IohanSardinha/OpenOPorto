import tkinter as tk
from tkinter import filedialog, messagebox
import geopandas as gpd
import numpy as np
from itertools import product, chain, combinations
from ipfn.ipfn import ipfn
from time import time
import pickle

class DimensionSelector(tk.Tk):
    def save(self):
        savedata = {
            "data":           self.data,
            "columns":        self.columns,
            "impossible":     self.impossible,
            "dim_selections": self.dim_selections,
            "section_div":    self.section_division_var.get(),
            "n_dims":         int(self.dim_var.get()),
        }
        filePath = filedialog.asksaveasfilename(title="Save state as", filetypes=[("Pickled dictionary",".pickle")])
        with open(filePath, "wb") as file:
            pickle.dump(savedata, file)

    def load(self):
         # 1. Load your saved state
        fp = filedialog.askopenfilename(
            title="Load state from", filetypes=[("Pickled dictionary", "*.pickle")]
        )
        if not fp:
            return
        try:
            with open(fp, "rb") as f:
                saved = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            return

        self.data            = saved["data"]
        self.columns         = saved["columns"]
        self.dim_selections  = saved["dim_selections"]
        saved_section        = saved.get("section_div", "None")
        saved_n_dims         = saved.get("n_dims", len(self.dim_selections))
        self.impossible      = saved["impossible"] if len(saved["impossible"]) and len(saved["impossible"][0]) == saved_n_dims else []

        # Apply number of dimensions *before* generating
        self.dim_var.set(saved_n_dims)

        # Trigger rebuild of dimensions & dropdown when data is present
        self.page2.pack_forget()
        self.page3.pack_forget()
        self.page1.pack(fill=tk.BOTH, expand=True)

        # Re‐populate Section Division options and set the saved one
        self.generate_section_dropdown()            # see next section
        self.section_division_var.set(saved_section)

        # Rebuild dimension listboxes (they’ll pick up self.dim_selections)
        self.generate_dimensions()
        self.update_listboxes()
        self.generate_section_dropdown()
        # If you’re already on page2 or page3, you can similarly call populate_forbid_listboxes()
            

    def __init__(self):
        super().__init__()
        self.title("INE Census IPF Synthetic Population Synthesizer")
        self.geometry("1100x800")
        
        self.data = None
        self.columns = []
        self.dimension_frames = []  # frames on page1
        self.listboxes = []         # listboxes for dimension selections (page1)
        self._updating = False      # flag to prevent recursive updates
        
        # These will store the current totals and selected columns for each dimension.
        self.sums = []
        self.dim_selections = []
        
        # Store impossible combinations as a list of tuples.
        self.impossible = []
        
        # This will hold the computed IPF matrix and CSV text.
        self.ipf_result = None
        self.csv_text = ""
        
        menuBar = tk.Menu(self, tearoff=False)
        self.config(menu=menuBar)

        fileOption = tk.Menu(menuBar,tearoff=False)
        menuBar.add_cascade(label="File", menu=fileOption, underline=0)

        fileOption.add_command(label="New",command=lambda:None, accelerator="Ctrl+n")
        fileOption.add_command(label="Save",command=self.save, accelerator="Ctrl+s")
        fileOption.add_command(label="Open",command=self.load, accelerator="Ctrl+o")

        # Create container frames for three pages.
        self.page1 = tk.Frame(self)
        self.page2 = tk.Frame(self)
        self.page3 = tk.Frame(self)
        
        # Create a StringVar for the spinbox and trace its changes.
        self.dim_var = tk.StringVar()
        self.dim_var.trace("w", self.on_dim_change)
        
        # Build the pages.
        self.build_page1()
        self.build_page2()
        self.build_page3()
        
        # Start with page1 visible.
        self.page1.pack(fill=tk.BOTH, expand=True)
    
    def generate_section_dropdown(self):
        # Clear existing menu
        menu = self.section_division_dropdown["menu"]
        menu.delete(0, "end")
        menu.add_command(label="None",
                        command=lambda: self.section_division_var.set("None"))
        # Add only those columns with unique values == number of rows
        nrows = len(self.data)
        for col in self.columns:
            if self.data[col].nunique() == nrows:
                menu.add_command(label=col,
                                command=lambda v=col: self.section_division_var.set(v))

    def build_page1(self):
        # --- File Selection ---
        file_frame = tk.Frame(self.page1)
        file_frame.pack(pady=10)
        load_button = tk.Button(file_frame, text="Load GeoPackage", command=self.load_file)
        load_button.pack()
        
        # --- Section Division Dropdown ---
        division_frame = tk.Frame(self.page1)
        division_frame.pack(pady=10)
        tk.Label(division_frame, text="Section Division:").pack(side=tk.LEFT)
        self.section_division_var = tk.StringVar(value="None")
        self.section_division_dropdown = tk.OptionMenu(
            division_frame, self.section_division_var, "None"
        )
        self.section_division_dropdown.pack(side=tk.LEFT, padx=5)
        
        # --- Dimension Count ---
        dim_frame = tk.Frame(self.page1)
        dim_frame.pack(pady=10)
        tk.Label(dim_frame, text="Enter number of dimensions (choose at least 2):").pack(side=tk.LEFT)
        self.dim_spinbox = tk.Spinbox(dim_frame, from_=2, to=10, width=5, textvariable=self.dim_var)
        self.dim_spinbox.pack(side=tk.LEFT, padx=5)
        
        # --- Container for Dimensions ---
        self.dimensions_container = tk.Frame(self.page1)
        self.dimensions_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- Status Label for Sum Consistency ---
        self.status_label = tk.Label(self.page1, text="", fg="red", font=("Arial", 12, "bold"))
        self.status_label.pack(pady=5)
        
        # --- Next Button ---
        next_button = tk.Button(self.page1, text="Next", command=self.goto_page2)
        next_button.pack(pady=10)

    
    def build_page2(self):
        # Page2: Impossible combinations.
        self.page2_inner = tk.Frame(self.page2)
        self.page2_inner.pack(pady=10, fill=tk.BOTH, expand=True)
        
        top_frame = tk.Frame(self.page2_inner)
        top_frame.pack(pady=10)
        tk.Label(top_frame, text="Select Impossible Combination", font=("Arial", 12, "bold")).pack()
        
        # Container for a listbox for each dimension.
        self.forbid_container = tk.Frame(self.page2_inner)
        self.forbid_container.pack(pady=10, fill=tk.X, expand=True)
        
        # Button to add impossible combinations.
        add_button = tk.Button(self.page2_inner, text="Add Combination(s)", command=self.add_impossible)
        add_button.pack(pady=5)
        
        # Display impossible combinations with a scrollbar.
        disp_frame = tk.Frame(self.page2_inner)
        disp_frame.pack(fill=tk.BOTH, padx=10, pady=5)
        tk.Label(disp_frame, text="Impossible Combinations:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.impossible_display = tk.Listbox(disp_frame, height=6, selectmode=tk.EXTENDED, exportselection=False)
        self.impossible_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        disp_scrollbar = tk.Scrollbar(disp_frame, orient="vertical", command=self.impossible_display.yview)
        disp_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.impossible_display.config(yscrollcommand=disp_scrollbar.set)
        
        # Button to remove selected impossible combinations.
        remove_button = tk.Button(self.page2_inner, text="Remove Selected Combination(s)", command=self.remove_impossible)
        remove_button.pack(pady=5)
        
        # Back and Compute IPF buttons.
        button_frame = tk.Frame(self.page2_inner)
        button_frame.pack(pady=10)
        back_button = tk.Button(button_frame, text="Back", command=self.goto_page1)
        back_button.pack(side=tk.LEFT, padx=10)
        compute_button = tk.Button(button_frame, text="Compute IPF", command=self.compute_ipf)
        compute_button.pack(side=tk.LEFT, padx=10)
    
    def build_page3(self):
        # Page3: Display results and allow saving CSV.
        top_frame = tk.Frame(self.page3)
        top_frame.pack(pady=10)
        tk.Label(top_frame, text="IPF Results (CSV Format)", font=("Arial", 12, "bold")).pack()
        
        # Create a Text widget with a vertical scrollbar.
        text_frame = tk.Frame(self.page3)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.result_text = tk.Text(text_frame, wrap="none")
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scrollbar = tk.Scrollbar(text_frame, orient="vertical", command=self.result_text.yview)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=text_scrollbar.set)
        
        # Save and Back buttons.
        button_frame = tk.Frame(self.page3)
        button_frame.pack(pady=10)
        back_button = tk.Button(button_frame, text="Back", command=self.goto_page2_from_page3)
        back_button.pack(side=tk.LEFT, padx=10)
        save_button = tk.Button(button_frame, text="Save CSV", command=self.save_csv)
        save_button.pack(side=tk.LEFT, padx=10)
    
    def on_dim_change(self, *args):
        if self.data is not None:
            self.generate_dimensions()
    
    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Select a GeoPackage file",
            filetypes=[("GeoPackage Files", "*.gpkg"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                self.data = gpd.read_file(file_path)
                self.columns = list(self.data.columns)
                messagebox.showinfo("File Loaded", f"Loaded file with {len(self.columns)} columns.")
                self.generate_dimensions()
                # Update the Section Division dropdown:
                # Find columns where the number of unique values equals the number of rows.
                candidates = [col for col in self.columns if len(self.data[col].unique()) == len(self.data)]
                menu = self.section_division_dropdown["menu"]
                menu.delete(0, "end")
                # Always add "None" as an option.
                menu.add_command(label="None", command=lambda: self.section_division_var.set("None"))
                for col in candidates:
                    menu.add_command(label=col, command=lambda value=col: self.section_division_var.set(value))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{e}")

    
    def generate_dimensions(self):
        # --- clear old frames/listboxes ---
        for frame in self.dimension_frames:
            frame.destroy()
        self.dimension_frames = []
        self.listboxes = []

        try:
            n_dims = int(self.dim_spinbox.get())
        except ValueError:
            return

        if not self.columns:
            return

        # For each dimension...
        for dim_idx in range(n_dims):
            frame = tk.Frame(self.dimensions_container, bd=2, relief=tk.SUNKEN)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            tk.Label(frame, text=f"Dimension {dim_idx+1}").pack()

            # --- Listbox + scrollbar ---
            lb_frame = tk.Frame(frame)
            lb_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            lb = tk.Listbox(lb_frame, selectmode=tk.EXTENDED, exportselection=False)
            lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            sb = tk.Scrollbar(lb_frame, orient="vertical", command=lb.yview)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            lb.config(yscrollcommand=sb.set)

            # Populate with all columns
            for col in self.columns:
                lb.insert(tk.END, col)

            # Bind update logic
            lb.bind("<ButtonRelease-1>", self.on_select)
            lb.bind("<KeyRelease>",      self.on_select)

            # Restore saved selections if available
            saved = getattr(self, 'dim_selections', None)
            if isinstance(saved, list) and len(saved) > dim_idx:
                lb.selection_clear(0, tk.END)                    # clear old
                for col_name in saved[dim_idx]:
                    if col_name in self.columns:
                        idx = self.columns.index(col_name)
                        lb.selection_set(idx)                     # select
                        lb.activate(idx)                          # focus
                # Trigger your color/total update
                lb.event_generate("<<ListboxSelect>>")

            # Total and Selected labels
            tot_lbl = tk.Label(frame, text="Total: 0", font=("Arial", 10, "bold"))
            tot_lbl.pack(pady=2)
            frame.total_label = tot_lbl

            sel_lbl = tk.Label(frame, text="Selected: None", font=("Arial", 10), wraplength=(frame.winfo_screenwidth()-700)/n_dims)
            sel_lbl.pack(pady=2)
            frame.selected_label = sel_lbl

            frame.listbox = lb
            self.dimension_frames.append(frame)
            self.listboxes.append(lb)
    
    def on_select(self, event):
        if not self._updating:
            self.after(100, self.update_listboxes)
    
    def update_listboxes(self):
        self._updating = True
        
        selections = []
        for lb in self.listboxes:
            selected = set(lb.get(i) for i in lb.curselection())
            selections.append(selected)
        
        for idx, lb in enumerate(self.listboxes):
            scroll_position = lb.yview()
            current_selection = selections[idx]
            other_selections = set()
            for j, sel in enumerate(selections):
                if j != idx:
                    other_selections |= sel
            
            allowed = current_selection | (set(self.columns) - other_selections)
            allowed_ordered = [col for col in self.columns if col in allowed]
            new_selection = current_selection.copy()
            
            lb.delete(0, tk.END)
            for col in allowed_ordered:
                lb.insert(tk.END, col)
                if col in new_selection:
                    lb.itemconfig(tk.END, bg="lightblue")
                else:
                    lb.itemconfig(tk.END, bg="white")
            
            for i, col in enumerate(allowed_ordered):
                if col in new_selection:
                    lb.selection_set(i)
            
            lb.yview_moveto(scroll_position[0])
        
        self.sums = []
        self.dim_selections = []
        for idx, frame in enumerate(self.dimension_frames):
            lb = self.listboxes[idx]
            selected_cols = [lb.get(i) for i in lb.curselection()]
            self.dim_selections.append(selected_cols)
            if self.data is not None and selected_cols:
                try:
                    total_val = self.data[selected_cols].sum().sum()
                except Exception:
                    total_val = 0
            else:
                total_val = 0
            self.sums.append(total_val)
            frame.total_label.config(text=f"Total: {total_val}")
            sel_text = ", ".join(selected_cols) if selected_cols else "None"
            frame.selected_label.config(text=f"Selected: {sel_text}")
        
        if self.sums and len(set(self.sums)) == 1 and all(self.dim_selections):
            for frame in self.dimension_frames:
                frame.total_label.config(fg="green")
            self.status_label.config(text="")
        else:
            for frame in self.dimension_frames:
                frame.total_label.config(fg="red")
            self.status_label.config(text="The sums must be equal and each dimension must have a selection", fg="red")
        
        self._updating = False
    
    def goto_page2(self):
        if not self.sums or len(set(self.sums)) != 1 or not all(self.dim_selections):
            messagebox.showwarning("Validation Error", "Ensure all dimensions have a selection and the sums are equal.")
            return
        self.page1.pack_forget()
        self.populate_forbid_listboxes()
        self.page2.pack(fill=tk.BOTH, expand=True)
    
    def populate_forbid_listboxes(self):
        # Clear old widgets
        for w in self.forbid_container.winfo_children():
            w.destroy()
        self.forbid_listboxes = []

        # Build one listbox per dimension
        for dim_idx, cols in enumerate(self.dim_selections):
            frame = tk.Frame(self.forbid_container, bd=2, relief=tk.SUNKEN)
            frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)
            tk.Label(frame, text=f"Dimension {dim_idx+1}").pack(pady=5)

            lbf = tk.Frame(frame)
            lbf.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            lb = tk.Listbox(lbf, selectmode=tk.EXTENDED, exportselection=False)
            lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            sb = tk.Scrollbar(lbf, orient="vertical", command=lb.yview)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            lb.config(yscrollcommand=sb.set)

            # Populate with only the columns *you selected* in page1
            for col in cols:
                lb.insert(tk.END, col)

            self.forbid_listboxes.append(lb)

        for comb in self.impossible:
            self.impossible_display.insert(tk.END, " | ".join(comb))

        
    def goto_page1(self):
        self.page2.pack_forget()
        self.page1.pack(fill=tk.BOTH, expand=True)
    
    def goto_page2_from_page3(self):
        # This method is used to return from the results page to page2.
        self.page3.pack_forget()
        self.page2.pack(fill=tk.BOTH, expand=True)
    
    def add_impossible(self, skip=False):
        selections = []
        for lb in self.forbid_listboxes:
            selected = [lb.get(i) for i in lb.curselection()]
            if not selected:
                messagebox.showwarning("Selection Missing", "Select at least one column in every dimension for the impossible combination.")
                return
            selections.append(selected)
        
        new_combinations = list(product(*selections))
        added_count = 0
        for comb in new_combinations:
            if comb not in self.impossible:
                self.impossible.append(comb)
                self.impossible_display.insert(tk.END, " | ".join(comb))
                added_count += 1
        if added_count == 0:
            messagebox.showinfo("No New Combination", "All selected combinations are already added.")
    
    def remove_impossible(self):
        selected_indices = list(self.impossible_display.curselection())
        if not selected_indices:
            messagebox.showwarning("No Selection", "Select impossible combination(s) to remove.")
            return
        for idx in sorted(selected_indices, reverse=True):
            self.impossible_display.delete(idx)
            del self.impossible[idx]
    
    def ipf_section(self, data):
        marginals = []
        for dim in self.dim_selections:
            marginals.append(data[dim].values)
        
        n_dims = len(self.dim_selections)
        
        if n_dims == 2:
            # Handle 2D case directly
            shape = tuple(len(sel) for sel in self.dim_selections)
            M = np.ones(shape, dtype=int)
            
            # Apply impossible combinations
            for forb in self.impossible:
                indices = []
                valid = True
                for d in range(n_dims):
                    try:
                        idx = self.dim_selections[d].index(forb[d])
                    except ValueError:
                        valid = False
                        break
                    indices.append(idx)
                if valid:
                    M[tuple(indices)] = 0
            
            # Run IPF with 1D marginals
            M = ipfn(M, marginals, [[0], [1]]).iteration()
        else:
            # Handle N-dimensional case where N > 2
            combs = list(combinations(range(n_dims), n_dims - 1))
            next_marginals = []
            next_dimensions = []
            
            for comb in combs:
                current_dims = comb
                current_marginals = [marginals[i] for i in current_dims]
                shape = [len(m) for m in current_marginals]
                sub_M = np.ones(shape)
                
                # Apply impossible combinations relevant to current_dims
                for forb in self.impossible:
                    indices = []
                    valid = True
                    for d in current_dims:
                        try:
                            idx = self.dim_selections[d].index(forb[d])
                        except ValueError:
                            valid = False
                            break
                        indices.append(idx)
                    if valid:
                        sub_M[tuple(indices)] = 0
                
                # Fit (N-1)-dimensional marginal using 1D marginals
                sub_layers = [[i] for i in range(len(current_dims))]
                sub_M = ipfn(sub_M, current_marginals, sub_layers).iteration()
                
                next_marginals.append(sub_M)
                next_dimensions.append(list(comb))  # Convert to list for ipfn compatibility
            
            # Initialize N-dimensional matrix
            shape = tuple(len(sel) for sel in self.dim_selections)
            M = np.ones(shape)
            
            # Apply all N-dimensional impossible combinations
            for forb in self.impossible:
                indices = []
                valid = True
                for d in range(n_dims):
                    try:
                        idx = self.dim_selections[d].index(forb[d])
                    except ValueError:
                        valid = False
                        break
                    indices.append(idx)
                if valid:
                    M[tuple(indices)] = 0
            
            # Prepare layers for N-dimensional IPF
            layers = [list(comb) for comb in combs]
            
            # Run final IPF
            M = ipfn(M, next_marginals, layers).iteration()
        
        return M


    def compute_ipf(self):
        
        sections = False
        if self.section_division_var.get() != "None":     
            sections = messagebox.askquestion("Section variable", "Section variable selected, compute IPF for each section?") == "yes"

        startTime = time()

        dims = [f"Var{i+1}" for i in range(len(self.dim_selections))]
        header = ",".join((["Section"] if sections else [])+dims + ["Value"])
        lines = [header]

        if sections:
            for _, row in self.data.iterrows():
                M = self.ipf_section(row)
        
                for index in np.ndindex(M.shape):
                    dims_values = [self.dim_selections[d][index[d]] for d in range(len(index))]
                    value = M[index]
                    line = ",".join([row[self.section_division_var.get()]]+dims_values + [str(value)])
                    lines.append(line)
        else:
            M = self.ipf_section(self.data[list(chain.from_iterable(self.dim_selections))].sum())
        
            for index in np.ndindex(M.shape):
                dims_values = [self.dim_selections[d][index[d]] for d in range(len(index))]
                value = M[index]
                line = ",".join(dims_values + [str(value)])
                lines.append(line)
        
        self.csv_text = "\n".join(lines)
        
        endTime = time()

        self.goto_page3()

        messagebox.showinfo("Computed IPF", f"Finished computing IPF in {endTime-startTime} seconds.\nResulted in {len(lines)-1} types of individuals")
    
    def goto_page3(self):
        self.page2.pack_forget()
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, self.csv_text)
        self.page3.pack(fill=tk.BOTH, expand=True)
    
    def save_csv(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                with open(file_path, "w") as f:
                    f.write(self.csv_text)
                messagebox.showinfo("Saved", f"CSV saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save CSV:\n{e}")
    
    def print_selections(self):
        for idx, lb in enumerate(self.listboxes):
            selected = [lb.get(i) for i in lb.curselection()]
            print(f"Dimension {idx+1}: {selected}")
        print("Impossible Combinations:")
        for comb in self.impossible:
            print(comb)

if __name__ == "__main__":
    app = DimensionSelector()
    app.mainloop()
