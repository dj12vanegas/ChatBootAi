from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel

# Cargar el modelo y el tokenizador preentrenados
model = GPT2LMHeadModel.from_pretrained('./results/model')
tokenizer = GPT2Tokenizer.from_pretrained('./results/tokenizer')

# Configurar el pipeline de generación con parámetros ajustados
generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    config={
        'max_length': 150,
        'temperature': 0.5,       # Reducir la temperatura para menos creatividad
        'top_p': 0.85,            # Ajustar top_p para mejorar la diversidad
        'top_k': 40,              # Ajustar top_k para controlar las opciones consideradas
        'repetition_penalty': 1.5 # Penalizar más la repetición
    }
)

def chat():
    print("¡Hola! Soy un chatbot con Inteligencia Artificial creado por DIEGO VANEGAS. ¿En qué puedo ayudarte hoy?")
    conversation_history = ""  # Mantener el historial de la conversación

    while True:
        try:
            user_input = input("Tú: ")
            if user_input.lower() in ['salir', 'exit', 'quit']:
                print("ChatBot: ¡Hasta luego!")
                break

            # Añadir la entrada del usuario al historial
            conversation_history += f"Tú: {user_input}\n"

            # Generar una respuesta usando el modelo preentrenado
            response = generator(conversation_history, max_length=150, num_return_sequences=1)
            generated_text = response[0]['generated_text'].strip()

            # Extraer la respuesta del chatbot y limpiar el historial
            bot_response = generated_text.split("\n")[-1]  # Obtener la última línea generada
            print(f"ChatBot: {bot_response}")

            # Añadir la respuesta del chatbot al historial
            conversation_history += f"ChatBot: {bot_response}\n"

        except (KeyboardInterrupt, EOFError, SystemExit):
            print("\nChatBot: ¡Adiós!")
            break

if __name__ == "__main__":
    chat()
