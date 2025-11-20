import pygame
import sys

pygame.init()
pygame.display.set_caption("Traffic Flow Prediction - UI Only")

WIDTH, HEIGHT = 700, 500
win = pygame.display.set_mode((WIDTH, HEIGHT))
FONT = pygame.font.SysFont("arial", 22)
BIG = pygame.font.SysFont("arial", 30, bold=True)

# ----------- UI COMPONENTS ---------------
class TextBox:
    def __init__(self, x, y, w, h, placeholder=""):
        self.rect = pygame.Rect(x, y, w, h)
        self.color_inactive = pygame.Color("gray50")
        self.color_active = pygame.Color("dodgerblue2")
        self.color = self.color_inactive
        self.text = ""
        self.placeholder = placeholder
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
            self.color = self.color_active if self.active else self.color_inactive

        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                if len(self.text) < 25:
                    self.text += event.unicode

    def draw(self, surf):
        pygame.draw.rect(surf, self.color, self.rect, 2)

        draw_text = self.text if self.text else self.placeholder
        color = "white" if self.text else "gray60"

        txt_surface = FONT.render(draw_text, True, color)
        surf.blit(txt_surface, (self.rect.x + 5, self.rect.y + 8))


class Button:
    def __init__(self, x, y, w, h, text):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text

    def draw(self, surf):
        pygame.draw.rect(surf, (70, 70, 70), self.rect)
        pygame.draw.rect(surf, (150, 150, 150), self.rect, 2)
        txt = FONT.render(self.text, True, (255, 255, 255))
        surf.blit(txt, (self.rect.x + (self.rect.w - txt.get_width()) // 2,
                        self.rect.y + (self.rect.h - txt.get_height()) // 2))

    def clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            return self.rect.collidepoint(event.pos)
        return False


# ------ Create input boxes ------
boxes = [
    TextBox(220, 150, 250, 40, "Site ID (e.g., 3002)"),
    TextBox(220, 210, 250, 40, "Direction (e.g. West-bound)"),
    TextBox(220, 270, 250, 40, "Start Time (e.g. 09:30)"),
    TextBox(220, 330, 250, 40, "End Time (e.g. 11:30)"),
    TextBox(220, 390, 250, 40, "Date (e.g. 05/10/2006)")
]


# ------ Buttons - EXACT 80px GAP ------

button_width = 120
button_height = 40
gap = 80

# Total width: 3 buttons + 2 gaps
total_width = button_width * 3 + gap * 2

start_x = (WIDTH - total_width) // 2
y_pos = 90

predict_btn = Button(start_x, y_pos, button_width, button_height, "Predict")
clear_btn   = Button(start_x + button_width + gap, y_pos, button_width, button_height, "Clear")
exit_btn    = Button(start_x + (button_width + gap) * 2, y_pos, button_width, button_height, "Exit")



# ---- Main loop ----
running = True
while running:
    win.fill((30, 30, 30))

    # Title
    title = BIG.render("Traffic Flow Prediction - Input Form", True, (255, 255, 255))
    win.blit(title, (WIDTH // 2 - title.get_width() // 2, 30))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        for box in boxes:
            box.handle_event(event)

        if predict_btn.clicked(event):
            print("Prediction requested:")
            print(f" Site: {boxes[0].text}")
            print(f" Dir: {boxes[1].text}")
            print(f" Start: {boxes[2].text}")
            print(f" End: {boxes[3].text}")
            print(f" Date: {boxes[4].text}")
            print("--------------------------------------")

        if clear_btn.clicked(event):
            for box in boxes:
                box.text = ""

        if exit_btn.clicked(event):
            running = False

    for box in boxes:
        box.draw(win)

    predict_btn.draw(win)
    clear_btn.draw(win)
    exit_btn.draw(win)

    pygame.display.update()

pygame.quit()
sys.exit()
