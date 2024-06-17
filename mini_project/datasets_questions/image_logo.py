from PIL import Image, ImageDraw, ImageFont

# Load the previous logo image
img_path = "/mnt/data/Enhance_the_previous_logo_for_'Personal_Evolution'.png"
img = Image.open(img_path)

# Define text properties
text_title = "Personal Evolution"
text_slogan = "Aim for the moon and harvest the stars"
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font_title = ImageFont.truetype(font_path, 30)  # Size for the title
font_slogan = ImageFont.truetype(font_path, 12)  # Size 12 for the slogan

# Create a draw object
draw = ImageDraw.Draw(img)

# Define positions for the text
title_position = (50, 50)  # Adjust based on the image
slogan_position = (50, 100)  # Adjust based on the image

# Add text to image
draw.text(title_position, text_title, font=font_title, fill="white")
draw.text(slogan_position, text_slogan, font=font_slogan, fill="white")

# Save the modified image
output_path = "/mnt/data/Enhanced_Logo_with_Slogan_Size_12.png"
img.save(output_path)

output_path
