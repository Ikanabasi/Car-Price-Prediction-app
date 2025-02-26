import streamlit as st
import pickle
import numpy as np
import pandas as pd

# @st.cache_resource
def load_model():
    with open('savedsteps.pkl', 'rb')as file:
        data = pickle.load(file)
    return data

data = load_model()

model_loaded = data['model']
preprocessor = data['preprocessor']

def show_predict_page():
    st.title('Car Price Prediction')

    st.write("""### We need some information to predict the price of the car""")


    transmission_type = ('Automatic', 'Manual')
    owner = ('First', 'Second', 'Unregistered')
    seller = ('Individual', 'Corporate')
    fuel = ('Petrol', 'Diesel')

    with st.form("car_price_form"):
        car_make = st.text_input("Enter car make")
        fuel_type = st.selectbox("Select fuel type", fuel)
        transmission = st.selectbox("Select transmission type", transmission_type)
        color = st.text_input("Enter color")
        owner_input = st.selectbox("Select owner", owner)
        sellertype = st.selectbox("Select sellertype", seller)
        engine = st.number_input("Engine (CC)")
        seating_capacity = st.number_input("Enter number of seats")
        fuel_tank_capacity = st.number_input("Enter fuel tank capacity")
        age = st.number_input("Enter age of car")

        ok = st.form_submit_button("Calculate price")

        if ok:

            if engine <= 0 or seating_capacity <= 0 or fuel_tank_capacity <= 0 or age <= 0:
                st.error("All numeric inputs must be greater than zero. Please correct your inputs.")
            else:

                a = np.array([[car_make, fuel_type, transmission, color, owner_input, sellertype,
                    engine, seating_capacity, fuel_tank_capacity, age]])

                columns = ['Make', 'Fuel Type', 'Transmission', 'Color', 'Owner', 'Seller Type',
                                'Engine', 'Seating Capacity', 'Fuel Tank Capacity', 'Age']

                a_df = pd.DataFrame(a, columns=columns)

                a_transformed = preprocessor.transform(a_df)

                price = model_loaded.predict(a_transformed)

                st.write(f"Predicted price is: ${price[0]:,.2f}")


