import pygame as p

class button():
    def __init__(self, x, y, image, scale):
        width = image.get_width()
        height = image.get_height()
        self.image = p.transform.scale(image, (int(width * scale), int(height * scale)))
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)
        self.clicked = False
        
    def draw(self, screen):
        action = False
        # get mouse position
        pos = p.mouse.get_pos()
        
        # check if the mouse is over the button
        if self.rect.collidepoint(pos):
            if p.mouse.get_pressed()[0] and not self.clicked:
                self.clicked = True
                action = True
        if p.mouse.get_pressed()[0] == 0:
            self.clicked = False
        # draw the button
        screen.blit(self.image, self.rect.topleft)
        return action

