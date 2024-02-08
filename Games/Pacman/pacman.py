from common import *
from hero import *
from enemy import *

import pygame
import sys


import numpy as np

# Initialize Pygame
pygame.init()


def draw_grid():
    for y, row in enumerate(consumed_grid):
        for x, cell in enumerate(row):
            rect = pygame.Rect(x * tile_size, y * tile_size, tile_size, tile_size)
            pygame.draw.rect(screen, value_to_color[cell], rect)

def prep_grid():
    pos = find_color_position('yellow')
    consumed_grid[pos.y, pos.x] = color_to_value['white']
    pos = find_color_position('red')
    consumed_grid[pos.y, pos.x] = color_to_value['white']

# Game loop flag
def main():
    prep_grid()
    pacman = Pacman()
    blinky = Ghost("blinky", pacman)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
        # Movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            pacman.set_direction(Direction.LEFT)
        if keys[pygame.K_RIGHT]:
            pacman.set_direction(Direction.RIGHT)
        if keys[pygame.K_UP]:
            pacman.set_direction(Direction.UP)
        if keys[pygame.K_DOWN]:
            pacman.set_direction(Direction.DOWN)

        # Screen update
        screen.fill(color_map['black'])
        draw_grid()
        pacman.draw()
        blinky.draw()
    
        pygame.display.flip()

        pacman.update()
        blinky.update()
        if pacman.isDead:
            print(f'score = {pacman.score}')
            running = False

        # Cap the frame rate
        pygame.time.Clock().tick(5)

    pygame.quit()
    sys.exit()

main()