import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.text import Text


#config is exported together witht the grid

##################CONFIG#############################
from matplotlib.colors import ListedColormap
nrows, ncols = 24, 24
colors = ['white', 'black', 'red', 'green', 'blue', 'yellow','orange', 'pink', 'teal', 'purple']
letters = [' ', 'A', 'B', 'C', 'D', 'E', 'r', 'p', 't', 'u']

cmap = ListedColormap(colors)
colors_len = len(colors)
brush_sizes = 5

use_exported = True
##################CONFIG#############################

if use_exported:
    import exported_grid 

is_drawing = False  # Track whether the mouse button is held down
brush_size = 1      # default brush size
brush_color = 0     # default brush color (white)
letter = 0          #default letter

fig, ax = plt.subplots()

color_axes = fig.add_axes([0.01, 0.5, 0.05, 0.45])  # Adjust as needed
brush_axes = fig.add_axes([0.07, 0.5, 0.05, 0.45])  # Adjusted left parameter
letter_axes = fig.add_axes([0.13, 0.5, 0.05, 0.45])  # Adjusted left parameter

# Example: Drawing color icons as rectangles
for i, color in enumerate(colors):
    color_axes.add_patch(Rectangle((0, i), 1, 1, color=color))

for i in range(brush_sizes):
    brush_axes.add_patch(Rectangle((0, i), 1, 1, color='lightgrey'))
    brush_axes.text(0.5, i + 0.5, str(i + 1), ha='center', va='center')

for i, l in enumerate(letters):
    letter_axes.add_patch(Rectangle((0, i), 1, 1, color='lightgrey'))
    letter_axes.text(0.5, i + 0.5, l, ha='center', va='center')

brush_axes.set_xlim(0, 1)
brush_axes.set_ylim(0, brush_sizes)
brush_axes.set_xticks([])
brush_axes.set_yticks([])

color_axes.set_xlim(0, 1)
color_axes.set_ylim(0, len(colors))
color_axes.set_xticks([])
color_axes.set_yticks([])

letter_axes.set_xlim(0, 1)
letter_axes.set_ylim(0, len(letters))
letter_axes.set_xticks([])
letter_axes.set_yticks([])

#np.array(np.random.choice(vals, (nrows, ncols)))
image = np.zeros((nrows, ncols), dtype=int) if not use_exported else np.array(exported_grid.color_grid)
letter_grid = np.zeros((nrows, ncols), dtype=int) if not use_exported else np.array(exported_grid.letter_grid) 

text_objects = [[ax.text(x, y, letters[letter_grid[y,x]], ha='center', va='center', fontweight='bold') if letter_grid[y,x] != 0 else None for x in range(ncols)] for y in range(nrows)]


mat = ax.matshow(image, cmap=cmap, vmin=0, vmax=colors_len)
ax.set_xticks(range(ncols))
ax.set_yticks(range(nrows))
ax.set_xticks(np.arange(-.5, ncols, 1), minor=True)
ax.set_yticks(np.arange(-.5, nrows, 1), minor=True)

ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
ax.tick_params(which='both', axis='both', length=0) # don't show tick marks


def on_release(event):
    global is_drawing
    is_drawing = False  # Stop drawing when the mouse button is released

def on_move(event):
    if is_drawing:
        draw(event)  # Continue drawing as the mouse moves

def onclick(event):

    global brush_color, brush_size, is_drawing, letter
    # Check if click is within the region for a color icon
    if event.inaxes == color_axes:
        if 0 <= event.xdata <= 1:  # Simplified check; adjust based on your layout
            brush_color = int(event.ydata)  # Simplified; map click y-coordinates to brush colors
            print(f'Brush Color = {brush_color}')

    if event.inaxes == brush_axes:
        if 0 <= event.xdata <= 1:  # Simplified check; adjust based on your layout
            brush_size= int(event.ydata) +1  # Simplified; map click y-coordinates to brush colors
            print(f'Brush size = {brush_size}')

    if event.inaxes == letter_axes:
        if 0 <= event.xdata <= 1:
            letter = int(event.ydata) 
            print(f'Letter = {letter}')

    if event.inaxes == ax:
        is_drawing = True
        draw(event)

def draw(event):
    global letter
    if event.inaxes == ax:
        x, y = int(event.xdata+0.5), int(event.ydata +0.5)

        for i in range(-brush_size + 1, brush_size):
                for j in range(-brush_size + 1, brush_size):
                    if 0 <= x + i < ncols and 0 <= y + j < nrows:
                        image[y + j, x + i] = brush_color
                        letter_grid[y + j, x + i] = letter  # Assign letter instead of int

        
                        if letter != ' ':
                        # Update or add new text
                            if text_objects[y + j][x + i] is not None:
                                text_objects[y + j][x + i].set_text(letters[letter])
                            else:
                                text_obj = ax.text(x + i, y + j, letters[letter], ha='center', va='center', fontweight='bold')
                                text_objects[y + j][x + i] = text_obj
                        else:
                            # Remove text if clearing the cell
                            if text_objects[y + j][x + i] is not None:
                                text_objects[y + j][x + i].set_visible(False)
                                text_objects[y + j][x + i] = None


        mat.set_array(image)
        plt.draw()    

def export_grid(image, letter_grid):
    with open('exported_grid.py', 'w') as file:
        # Export configuration settings
        file.write(f"# Configuration\n")
        file.write(f"nrows, ncols = {nrows}, {ncols}\n")
        file.write(f"colors = {colors}\n")
        file.write(f"brush_sizes = {brush_sizes}\n")
        file.write(f"letters_list = {letters}\n") # Rename to avoid conflict with 'letters' grid variable
        file.write("\n")

        # Export color grid
        file.write('color_grid = [\n')
        for row in image:
            file.write('    ' + str(list(row)) + ',\n')
        file.write(']\n\n')

        # Export letter grid
        file.write('letter_grid = [\n')
        for row in letter_grid:
            file.write('    ' + str(list(row)) + ',\n')
        file.write(']\n\n')
         
        file.write('text_grid = [\n')
        for row in letter_grid:
            letter_row = [letters[index] for index in list(row)]
            file.write('    ' + str(list(letter_row)) + ',\n')

        file.write(']\n')
    
    fig.savefig('exported_grid.png')
    print("Grid and configuration exported to exported_grid.py and exported_grid.png")

def on_close(event):
    export_grid(image, letter_grid)
    print("exported_grid.py has been created with the current grid values.")
    print("color grid\n", image)
    print("letter grid\n", letter_grid)

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_move)
fig.canvas.mpl_connect('close_event', on_close)

plt.show()

