from utils import detect_hand
import cv2
import os
import numpy as np
import urllib.request
from config.config import BOT_CONFIG

import telebot

result_storage_path = 'tmp'

texts = ['Подбираем легушьку для вас...', 'Жабоньки уже в пути!', 'Отправляемся в болото...', 'Квакусики рядом!',
         'Ожидаем лягуху...']


def save_image(message):
    cid = message.chat.id
    image_id = get_image_id_from_message(message)
    bot.send_message(cid, np.random.choice(texts))
    file_path = bot.get_file(image_id).file_path
    image_url = "https://api.telegram.org/file/bot{0}/{1}".format(token, file_path)
    if not os.path.exists(result_storage_path):
        os.makedirs(result_storage_path)
    image_name = "{0}.jpg".format(image_id)
    urllib.request.urlretrieve(image_url, "{0}/{1}".format(result_storage_path, image_name))

    return image_name


def get_image_id_from_message(message):
    return message.photo[len(message.photo) - 1].file_id


def image_processing(image_name):
    image_np = cv2.imread('./tmp/' + image_name)
    toad = cv2.imread('./data/toads/' + np.random.choice(os.listdir('./data/toads/')), cv2.IMREAD_UNCHANGED)
    best_image, num_boxes = detect_hand.predict_and_draw(image_np, toad)
    cv2.imwrite(image_name, best_image)
    return num_boxes


token = BOT_CONFIG["token"]
bot = telebot.TeleBot(token)


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == "/help":
        bot.send_message(message.from_user.id, "Отправь мне фотографию ладошки!")
    elif message.text == "/start":
        bot.send_message(message.from_user.id, "Жду фотографию пустой ладошки...")


@bot.message_handler(content_types=['photo'])
def get_photo_message(message):
    img_name = save_image(message)
    num_boxes = image_processing(img_name)
    if num_boxes == 0:
        bot.send_message(message.from_user.id, "Хмммм, не получилось задетектировать ладонь. Может, отправишь ещё одно фото или исправить освещение?")
    else:
        bot.send_photo(message.chat.id, open(img_name, 'rb'))


bot.polling(none_stop=True, interval=0)
