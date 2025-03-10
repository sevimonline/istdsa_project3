import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Sayfa Ayarları
st.set_page_config(
    page_title="House Classifier",
    page_icon="https://miro.medium.com/v2/resize:fit:2400/1*rGi8_JUoGX0L3W6nivmIAg@2x.png",
    menu_items={
        "Get help": "mailto:berke.sevim@istdsa.com",
        "About": "For More Information\n" + "https://github.com/istdsa"
    }
)

# Başlık Ekleme
st.title("House Classification Project")

# Markdown Oluşturma
st.markdown("A research company wants to decide whether this house is in **:red[New York City]** or **:red[San Francisco]** by looking at the various features of the houses they have.")

# Resim Ekleme
st.image("https://i.insider.com/5808fc6cc52402ce248b5aa2?width=1000&format=jpeg&auto=webp")

st.markdown("After the latest developments in the artificial intelligence industry, they expect us to develop a **machine learning model** in line with their needs and help them with their research.")
st.markdown("In addition, when they have information about a new house, they want us to come up with a product that we can predict where this house will be based on this information.")
st.markdown("*Let's help them!*")

st.image("https://resources.pollfish.com/wp-content/uploads/2020/11/MARKET_RESEARCH_FOR_REAL_ESTATE_IN_CONTENT_1.png")

# Header Ekleme
st.header("Data Dictionary")

st.markdown("- **in_sf**: Where is the house? (0 = New York City, 1 = San Francisco)")
st.markdown("- **beds**: Number of beds in the house")
st.markdown("- **bath**: Number of bath in the house")
st.markdown("- **price**: Sale price of the house")
st.markdown("- **year_built**: Year of the build of the house")
st.markdown("- **sqft**: Square-feet of the house")
st.markdown("- **price_per_sqft**: Price per square feet of the house")
st.markdown("- **elevation**: Elevation(ft) value where the house is located")

# Pandasla veri setini okuyalım
df = pd.read_pickle("train_df.pkl")

# Küçük bir düzenleme :)
df.beds = df.beds.astype(int)
df.bath = df.bath.astype(int)

# Tablo Ekleme
st.table(df.sample(5))

#---------------------------------------------------------------------------------------------------------------------

# Sidebarda Markdown Oluşturma
st.sidebar.markdown("**Choose** the features below to see the result!")

# Sidebarda Kullanıcıdan Girdileri Alma
name = st.sidebar.text_input("Name", help="Please capitalize the first letter of your name!")
surname = st.sidebar.text_input("Surname", help="Please capitalize the first letter of your surname!")
price = st.sidebar.number_input("Price of House ($)", min_value=1, format="%d")
sqft = st.sidebar.number_input("Square Feet of House", min_value=1)
price_per_sqft = price/sqft
elevation = st.sidebar.slider("Elevation of House (ft)", min_value=0, max_value=250)

#---------------------------------------------------------------------------------------------------------------------

# Pickle kütüphanesi kullanarak eğitilen modelin tekrardan kullanılması
from joblib import load

logreg_model = load('logreg_model.pkl')

input_df = pd.DataFrame({
    'elevation': [elevation],
    'price_per_sqft': [price_per_sqft]
})

# Tahminlerimizin toplantı dataframe oluşturma
train_df = df[['elevation','price_per_sqft']]
result_df = pd.concat([train_df, input_df])

# Verilerimizi ölçeklendirmeyi unutmuyoruz!
std_scale = StandardScaler()
scaled_result_df = std_scale.fit_transform(result_df)

# Kullanıcı tarafından girilen girdiye ulaşma
test_df = scaled_result_df[-1].reshape(1,2) 

pred = logreg_model.predict(test_df)
pred_probability = np.round(logreg_model.predict_proba(test_df), 2)

#---------------------------------------------------------------------------------------------------------------------

st.header("Results")

# Sonuç Ekranı
if st.sidebar.button("Submit"):

    # Info mesajı oluşturma
    st.info("You can find the result below.")

    # Sorgulama zamanına ilişkin bilgileri elde etme
    from datetime import date, datetime

    today = date.today()
    time = datetime.now().strftime("%H:%M:%S")

    # Sonuçları Görüntülemek için DataFrame
    results_df = pd.DataFrame({
    'Name': [name],
    'Surname': [surname],
    'Date': [today],
    'Time': [time],
    'Elevation': [elevation],
    'Price': [price],
    'Sqft': [sqft],
    'Prediction': [pred],
    'NY Probability': [pred_probability[0,0]],
    'SF Probability': [pred_probability[0,1]]
    })

    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("0","NY"))
    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("1","SF"))

    st.table(results_df)

    if pred == 0:
        st.image("https://images.vexels.com/media/users/3/144125/isolated/preview/e41e827336e592fc084566be2bff2665-new-york-skyline-badge-vector.png")
    else:
        st.image("https://images.vexels.com/media/users/3/144138/isolated/preview/e69d5f1721fe4a3c0afa93679f4d944f-san-francisco-skyline-badge.png")
else:
    st.markdown("Please click the *Submit Button*!")