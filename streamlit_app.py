import streamlit as st
from dotenv import load_dotenv
from main_chatbot import modelo_final  # Asegúrate de que tu función está en un script accesible

# Cargar variables de entorno
load_dotenv()

# Título de la página
st.title('SuperBot')
st.markdown("""
<style>
.subheader {
    font-size: 16px; /* Ajusta el tamaño de la fuente según necesites */
    font-weight: normal; /* Esto hace que la letra no sea negrita */
}
</style>
<p class="subheader">Inteligencia artificial para compras más inteligentes</p>
""", unsafe_allow_html=True)
# Crear dos columnas para los logos
col1, col2 = st.columns(2)

# Logo de Mercadona en la primera columna
col1.image("https://upload.wikimedia.org/wikipedia/commons/9/90/Logo_Mercadona_%28color-300-alpha%29.png", width=250)

# Logo de DIA en la segunda columna
col2.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Dia_Logo.svg/1200px-Dia_Logo.svg.png", width=150)

# Subtítulo y descripción
st.markdown("""
### Encuentra las mejores ofertas y compara productos entre Mercadona y DIA
Utiliza nuestro chatbot para obtener información detallada y actualizada sobre productos, comparar precios y mucho más. Simplemente escribe tu consulta en el campo de texto y descubre cómo puedes ahorrar en tu próxima compra.
""")

# Input del usuario
input_usuario = st.text_input("Escribe tu pregunta aquí:")

# Botón para enviar la pregunta
if st.button('Enviar'):
    if input_usuario:
        # Llamar a la función del chatbot y obtener la respuesta
        respuesta = modelo_final(input_usuario)
        st.write("Respuesta:", respuesta)
    else:
        st.error("Por favor, ingresa una pregunta.")


