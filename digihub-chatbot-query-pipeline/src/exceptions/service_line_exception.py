
class UnAuthorizedServiceLineException(Exception):
    def __init__(self, message="Query Belongs To UnAuthorized Service Line",disclaimar=""):
        super().__init__(message)
class OutOfScopeException(Exception):
    def __init__(self, message="Let’s keep it work-related.\n\nI’m here to help with questions about SITA documentation and your available services. For anything else, I recommend checking your favorite search engine outside of DigiHub.",
                 final_response="",detected_language="",auto_msg=False):
        if auto_msg:
            OUT_OF_SCOPE_MESSAGES = {
                "english": "Let’s keep it work-related.\n\nI’m here to help with questions about SITA documentation and your available services. For anything else, I recommend checking your favorite search engine outside of DigiHub.",
                "german": "Ich unterstütze Sie gerne bei Fragen zur SITA-Dokumentation und zu den für Sie verfügbaren Diensten. Für alles andere empfehle ich, eine Suchmaschine außerhalb von DigiHub zu nutzen",
                "french": "Je suis à votre disposition pour vous accompagner sur la base des documents SITA et des services auxquels vous avez access. Pour toute autre demande en dehors de ce cadre, je vous recommande d'utiliser votre moteur de recherche habituel.",
                "spanish": "Estoy aquí para ayudarte con las preguntas que tengas acerca de la documentación de SITA y los servicios disponibles. Para cualquier otro tema, te recomiendo usar tu buscador habitual fuera de DigiHub."
            }

            message = OUT_OF_SCOPE_MESSAGES.get(detected_language.lower(), OUT_OF_SCOPE_MESSAGES["english"])
        super().__init__(message)

class PartialAccessServiceLineException(Exception):
    def __init__(self, message="Query Belongs To UnAuthorized Service Line",disclaimar=""):
        super().__init__(message,disclaimar)
