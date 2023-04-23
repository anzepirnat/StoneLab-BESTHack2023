import streamlit as st
import os
import cv2
import numpy as np
# text recognition
import pytesseract
import numpy as np 
import pandas as pd 
#import sklearn 
import os
import difflib
import re
import spacy
import pickle
import datetime
from collections import defaultdict
import plotly.graph_objs as go
import altair as alt
import matplotlib.pyplot as plt
#import h5py

st.set_page_config(page_title="AI CO2", page_icon="🤖")


def OnlyNouns(text):
    nlp = spacy.load("en_core_web_sm")
    noun = []
    doc = nlp(text)
    for token in doc:
        if token.pos_ == 'PROPN':
            noun.append(token.text)
    return noun

def AltChart(data, time_window):
    
    time_window_idx = [3,6,9,12,15,18,21,24]

    hidx = 0

    for idx in range(len(time_window_idx)):
        if time_window_idx[idx] == time_window:
            hidx = idx

    labels_order = data[1]
    on_off = data[0][hidx]

    lst = []

    lst.append(on_off)
    lst.append(labels_order)

    df = pd.DataFrame({
        'appliance:N': labels_order,
        'count(on_off):Q': on_off
    })

    st.dataframe(df)

    chart = alt.Chart(df).mark_bar(size=20).encode(
    x=alt.X('appliance:N', axis=alt.Axis(title='Appliances'), sort=labels_order),
    y=alt.Y('count(on_off):Q', axis=alt.Axis(title='1=ON, 0=OFF')),
    ).properties(
    width=600,
    height=400
    )

    return chart

def plot_bar(data, time_window):
    time_window_idx = [3, 6, 9, 12, 15, 18, 21, 24]

    hidx = 0
    for idx in range(len(time_window_idx)):
        if time_window_idx[idx] == time_window:
            hidx = idx

    labels_order = data[1]
    on_off = data[0][hidx]

    # Create the plot
    fig, ax = plt.subplots(figsize=(15,10))
    ax.bar(labels_order, on_off)

    # Set the labels and title
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['OFF', 'ON'])
    ax.set_xlabel('Appliances')
    ax.set_ylabel('1=ON, 0=OFF')
    ax.set_title('Bar Plot')

    # Show the plot in Streamlit
    st.pyplot(fig)

#def izdelek_iz_baze(poved, tabela_izdelkov):
#    regex = re.compile('[^a-zA-Z]')
#    hrana = tabela_izdelkov[0]
#    besede = poved.split()
#
#
#    bestmatch = []
#    for beseda in besede:
#        for izdelek in hrana:
#            for deli in izdelek.split():
#                beseda = regex.sub('', beseda).lower()
#                deli = regex.sub('', deli).lower()
#                s = difflib.SequenceMatcher(None, deli, beseda)
#                if beseda != '':
#                    bestmatch.append({'izdelek':izdelek,'beseda':beseda, 'del': deli, 'value':s.ratio() })
#    bestmatch.sort(reverse = True, key = lambda element: element['value'])
#
#    zmagovalci = []
#    for zmagovalec in bestmatch:
#        if zmagovalec['value'] == bestmatch[0]['value']:
#            zmagovalci.append(zmagovalec)
#
#    stevilo = 10
#    value = 0
#    true_winner = {}
#    for zmagovalec in zmagovalci:
#        n = len(zmagovalec['del']) - len(zmagovalec['beseda'])
#        if n < stevilo:
#            stevilo = n
#            true_winner = zmagovalec
#            value = zmagovalec['value']
#    if value > 0.59:
#        izdelek = true_winner['izdelek']
#        return [izdelek, tabela_izdelkov[1][hrana.index(izdelek)]]
#    else:
#        return []
#    
def izdelek_iz_baze(tabela, tabela_izdelkov):
    vse = []
    tabela.reverse()
    prejsnja = ''
    for poved in tabela:
        regex = re.compile('[^a-zA-Z]')
        hrana = tabela_izdelkov[0]
        besede = poved.split()
        bestmatch = []
        for beseda in besede:
            for izdelek in hrana:
                for deli in izdelek.split():
                    nova = regex.sub('', beseda).lower()
                    deli = regex.sub('', deli).lower()
                    if ((deli in nova) or (nova in deli)) and (nova != '') and (deli != ''):
                        n = len(nova) - len(deli)
                        bestmatch.append({'izdelek': izdelek,'beseda': beseda, 'del': deli, 'razlika': n, 'prejsnja': prejsnja})
        if poved != '':
            prejsnja = poved
        if bestmatch != []:
            bestmatch.sort(reverse = True, key = lambda element: element['razlika'])
            zmagovalec = bestmatch[0]
            try:
                index = zmagovalec['prejsnja'].index('k')
                stevilo = zmagovalec['prejsnja'][re.search(r"\d", zmagovalec['prejsnja']).start():index]
                stevilo = float(stevilo)
                vse.append([zmagovalec['izdelek'], stevilo*tabela_izdelkov[1][hrana.index(zmagovalec['izdelek'])]])#]])
            except:
                vse.append([zmagovalec['izdelek'],tabela_izdelkov[1][hrana.index(zmagovalec['izdelek'])]])
    return vse

def save_image(img, filename):
    cv2.imwrite(filename, img)

def read_qr_code(filename):
    """Read an image and read the QR code.
    
    Args:
        filename (string): Path to file
    
    Returns:
        qr (string): Value from QR code
    """
    
    try:
        img = cv2.imread(filename)
        detect = cv2.QRCodeDetector()
        value, points, straight_qrcode = detect.detectAndDecode(img)
        return value
    except:
        return st.error("šajze paradajze, ni uspel")

def take_image():
    #st.title("Take a picture and save it to a directory")
    save_directory = "."

    # Check if the directory exists, if not create it
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Use OpenCV to capture a picture from the camera
    cap = cv2.VideoCapture(0)
    #st.image(cap.read()[1], channels="BGR")

    _, img = cap.read()

    # Resize the captured image
    scale_percent = 50
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=3)
    sobel = cv2.addWeighted(sobelx, 0.8, sobely, 0.2, 0)

    st.image(sobel)
    

    filename = os.path.join(save_directory, "image.jpg")
    save_image(img, filename)
    #st.success("Image saved successfully")

    return sobel

def img_to_txt():
    img = cv2.imread('image.jpg')

    # configurations
    config = ('-l eng --oem 1 --psm 3')

    # pytesseract path
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    text = pytesseract.image_to_string(img, config=config)

    # print results
    text = text.split('\n')
    print(text)

    return text

def img_to_txt_2(img_file):
    # Convert the file object to a numpy array
    img_array = np.frombuffer(img_file.read(), np.uint8)
    # Decode the numpy array to an image
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # configurations
    config = ('-l eng --oem 1 --psm 3')

    # pytesseract path
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    text = pytesseract.image_to_string(img, config=config)

    # print results
    text = text.split('\n')
    print(text)

    return text

########################################################################
########################################################################
########################################################################
########################################################################
########################################################################

time = datetime.time(22, 48)
date = datetime.date(2015, 7, 17)

text_storage = [

]

###################### TRANSPORT ##########################
transport = [

]

####################### FOOD DEPARTMENT ####################
path = './data/food/GHG-emissions-by-life-cycle-stage-OurWorldinData-upload.csv'  
df = pd.read_csv(path)

food = [
    [],
    []
]

final_food = pickle.load(open('./data/food/final_food.pkl', 'rb'))
#st.markdown(final_food)

for i in range(len(df)):
    vrstica = df.iloc[i]
    food[0].append(vrstica['Food product'])
    suma = vrstica['Land use change'] + vrstica['Animal Feed'] + vrstica['Farm'] + vrstica['Processing'] + vrstica['Transport'] + vrstica['Packging'] + vrstica['Retail']
    food[1].append(suma)

food[0].append('Cappuccino')
food[1].append(9.65)
food[0].append('Americano')
food[1].append(16.5)

# Using object notation
mode = st.sidebar.selectbox(
    "Select an action",
    ("Scan receipt 🧾", "Track appliance activity 🔍", "Carbon footprint overview 🦶🏿")
)

if mode == 'Scan receipt 🧾':

    st.header('Scan receipt')

    option = st.radio("You can either take a picture of receit using camera or upload an existing picture from storage.", 
        ('Use camera 📷', 'Upload from device 🔼')
    )

    if option == 'Use camera 📷':
        if st.button("Press when ready"):
            
            img = take_image()
            #st.image(img)
            text = img_to_txt()
            #text = OnlyNouns(text)
            text_storage.append(text)
            #st.markdown(text)

            cnter = 0
            if izdelek_iz_baze(text, food):
                #st.markdown(izdelek_iz_baze(text, food))
                for izdelek in izdelek_iz_baze(text,food):
                    final_food.append([izdelek[0], izdelek[1], time, date])
                    cnter += 1
            
            #st.markdown(final_food)

            pickle.dump(final_food, open('./data/food/final_food.pkl', 'wb'))

            count = 0
            for f in range(cnter):
                count += final_food[len(final_food)-cnter-1+f][1]
            st.markdown('Your carbon footprint is:')
            st.subheader(f'{count} kg CO2 eq. / kg')

    if option == 'Upload from device 🔼':
        img = st.file_uploader("Choose a file")

        if st.button("Scan uploaded image"):
            st.image(img)
            text = img_to_txt_2(img)
            #st.text(text)
            #text = OnlyNouns(text)
            cnter = 0
            if izdelek_iz_baze(text, food):
                #st.markdown(h)
                #st.markdown(izdelek_iz_baze(text, food))
                for izdelek in izdelek_iz_baze(text, food):
                    final_food.append([izdelek[0], izdelek[1], time, date])
                    cnter += 1

            
            # Save the updated table to a CSV file
            pickle.dump(final_food, open('./data/food/final_food.pkl', 'wb'))
            count = 0
            for f in range(cnter):
                count += final_food[len(final_food)-cnter-1+f][1]
            st.markdown('Your carbon footprint is:')
            st.subheader(f'{count} kg CO2 eq. / kg')






if mode == 'Track appliance activity 🔍':
    st.header("Appliance activity")
    time_window = st.slider('Set the length of the time window you wish to view', 3, 24, step=3)

    data = pickle.load(open('./data/NILM/device_activity.pkl', 'rb'))

    if time_window == 3:
        st.metric('',f'{time_window}h long time window')
        plot_bar(data,time_window)
    if time_window == 6:
        st.metric('',f'{time_window}h long time window')
        plot_bar(data,time_window)
    if time_window == 9:
        st.metric('',f'{time_window}h long time window')
        plot_bar(data,time_window)
    if time_window == 12:
        st.metric('',f'{time_window}h long time window')
        plot_bar(data,time_window)
    if time_window == 15:
        st.metric('',f'{time_window}h long time window')
        plot_bar(data,time_window)
    if time_window == 18:
        st.metric('',f'{time_window}h long time window')
        plot_bar(data,time_window)
    if time_window == 21:
        st.metric('',f'{time_window}h long time window')
        plot_bar(data,time_window)
    if time_window == 24:
        st.metric('',f'{time_window}h long time window')
        plot_bar(data,time_window)







if mode == 'Carbon footprint overview 🦶🏿':
    st.header("Carbon footprint")

    tab1, tab2, tab3 = st.tabs(["🍞 Food", "⚡ Electricity", "🚗 Transport"])
  
    ################################# tab1 #################################
    tab1.subheader("Week chart")
    count_by_day = []
    counts = defaultdict(int)

    for food in final_food:
        date = food[3]
        count = food[1]
        counts[date] += count

    for date, total_count in counts.items():
        count_by_day.append((date, total_count))
        
    count_by_day.sort()
    #print(count_by_day)
    #tab1.markdown(count_by_day)

    # create a pandas dataframe from count_by_day
    df = pd.DataFrame(count_by_day, columns=['date', 'count'])

    df['date'] = pd.to_datetime(df['date'])

    df['day_of_week'] = df['date'].dt.day_name()

    # Define the order of days of the week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Create a bar chart using Altair
    chart = alt.Chart(df).mark_bar(size=20).encode(
    x=alt.X('day_of_week:N', axis=alt.Axis(title='Day of the week'), sort=day_order),
    y=alt.Y('count:Q', axis=alt.Axis(title='Carbon footprint [kg of CO2 eq.]')),
    ).properties(
    width=600,
    height=400
    )

    # Display the chart using Streamlit
    tab1.altair_chart(chart)

    tab1.subheader('Improve fast by consuming less:')

    # Sort the list by the second element in descending order
    final_food_sorted = sorted(final_food, key=lambda x: x[1], reverse=True)

    final_food_sorted = sorted(final_food, key=lambda x: x[1], reverse=True)

    # Get the first and second element of the three sublists with the highest second element
    top3_food = []
    seen_items = set()
    for item in final_food_sorted:
        if item[0] not in seen_items:
            top3_food.append([item[0], item[1]])
            seen_items.add(item[0])
            if len(top3_food) == 3:
                break

    #tab1.markdown(top3_food)

    df = pd.DataFrame(top3_food, columns=['Food', 'kg of CO2 eq.']).reset_index(drop=True)
    df.index = df.index + 1
    tab1.table(df)





    ################################# tab2 #################################
    tab2.subheader("Electricity")
    #f = h5py.File('./data/NILM/refit.hdf5', 'r')

    #alldata = pickle.load(open('./data/NILM/PEH2/PEH2_24h.pkl', 'rb'))

    #x_train = alldata[9][0]

    #x = x_train[4].flatten().tolist()


    alldata = pd.read_csv('./data/NILM/household_power_consumption.csv')
    alldata = alldata.values.tolist()

    opt = tab2.radio("Do you want to see your consumption for the past day, past week or past month", ("1 day", "1 week", "1 month"))



    if opt == "1 day":
        dayworth = []
        for i in range(1400):
            podatek = float(alldata[i][2])
            dayworth.append(podatek*1000)

        max_day_val, max_day_idx = max((val, idx) for idx, val in enumerate(dayworth))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dayworth)
        ax.set_xticks(range(0, 1440, 60))
        ax.set_xticklabels([f"{hour}h" for hour in range(1, 25)])
        tab2.pyplot(fig)

        tab2.subheader('Consumption report:')
        rough_hour = max_day_idx // 60
        tab2.text(f"At {rough_hour} o'clock your consumption was highest {max_day_val}")

    if opt == "1 week":
        dayworth = []
        for i in range(1400*7):
            podatek = float(alldata[i][2])
            dayworth.append(podatek*1000)

        max_week_val, max_week_idx = max((val, idx) for idx, val in enumerate(dayworth))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dayworth)
        ax.set_xticks(range(0, 1440*7, 1440))
        ax.set_xticklabels(['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'])
        tab2.pyplot(fig)

        tab2.subheader('Consumption report:')
        days = ['monday', 'tuesday', 'wednsday', 'thursday', 'friday', 'saturday', 'sunday']
        rough_day = max_week_idx // 1440
        rough_day = days[rough_day]
        tab2.text(f"On {rough_day} your consumption was highest at {round(max_week_val,2)} W")

    if opt == "1 month":
        dayworth = []
        for i in range(1400*31):
            if alldata[i][2] != '?':
                podatek = float(alldata[i][2])
            dayworth.append(podatek*1000)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dayworth)
        ax.set_xticks(range(0, 1440*31, 1440))
        ax.set_xticklabels([f"{day}" for day in range(1, 32)])
        tab2.pyplot(fig)

    ################################# tab3 #################################
    tab3.subheader("Transport")
    # insert valerijev shit


#st.header(final_food)