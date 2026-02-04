import re

import random
from typing import Optional

# Place this at the top of the file or in a new utility file
KEYWORD_AWARE_MESSAGES = {
    "english": [
        "I see you're asking about {keyword}. I don't have specific information on that in my knowledge base. Could you try rephrasing your question with more details? It would help me find what you're looking for.",
        "You mentioned {keyword}. To give you the most accurate answer, I need a bit more context. Could you please provide more details about your query?",
        "Thanks for your question about {keyword}. I couldn't find a direct answer. Try asking in a different way, perhaps by focusing on a specific process or document.",
        "While I can't find specific details on your query about {keyword}, I can help with information found in SITA's service documentation. Could you perhaps narrow down your question?"
    ],
    "german": [
        "Ich sehe, Sie fragen nach {keyword}. In meiner Wissensdatenbank habe ich dazu keine spezifischen Informationen. Könnten Sie Ihre Frage bitte mit mehr Details umformulieren? Das würde mir helfen, das zu finden, was Sie suchen.",
        "Sie haben {keyword} erwähnt. Um Ihnen die genaueste Antwort geben zu können, benötige ich etwas mehr Kontext. Könnten Sie bitte weitere Details zu Ihrer Anfrage angeben?",
        "Vielen Dank für Ihre Frage zu {keyword}. Ich konnte keine direkte Antwort finden. Versuchen Sie, die Frage anders zu formulieren, vielleicht indem Sie sich auf einen bestimmten Prozess oder ein Dokument konzentrieren.",
        "Obwohl ich keine spezifischen Details zu Ihrer Anfrage über {keyword} finden kann, kann ich Ihnen mit Informationen aus der SITA-Servicedokumentation helfen. Könnten Sie Ihre Frage vielleicht eingrenzen?"
    ],
    "french": [
        "Je vois que votre question concerne {keyword}. Je n'ai pas d'informations spécifiques à ce sujet dans ma base de connaissances. Pourriez-vous reformuler votre question avec plus de détails pour m'aider à trouver ce que vous cherchez ?",
        "Vous avez mentionné {keyword}. Pour vous donner la réponse la plus précise, j'aurais besoin d'un peu plus de contexte. Pouvez-vous fournir plus de détails sur votre demande ?",
        "Merci pour votre question sur {keyword}. Je n'ai pas trouvé de réponse directe. Essayez de poser votre question différemment, peut-être en vous concentrant sur un processus ou un document spécifique.",
        "Bien que je ne trouve pas de détails spécifiques sur votre demande concernant {keyword}, je peux vous aider avec les informations contenues dans la documentation des services SITA. Pourriez-vous préciser votre question ?"
    ],
    "spanish": [
        "Veo que preguntas sobre {keyword}. No tengo información específica sobre eso en mi base de conocimientos. ¿Podrías intentar reformular tu pregunta con más detalles? Me ayudaría a encontrar lo que buscas.",
        "Has mencionado {keyword}. Para darte la respuesta más precisa, necesito un poco más de contexto. ¿Podrías proporcionar más detalles sobre tu consulta?",
        "Gracias por tu pregunta sobre {keyword}. No he podido encontrar una respuesta directa. Intenta preguntar de una manera diferente, quizás centrándote en un proceso o documento específico.",
        "Aunque no encuentro detalles específicos sobre tu consulta acerca de {keyword}, puedo ayudarte con la información que se encuentra en la documentación de los servicios de SITA. ¿Podrías acotar un poco más tu pregunta?"
    ]
}

KEYWORDS = ["DigiHub", "SITA"]

def get_keyword_aware_message(query: str, language: str) -> Optional[str]:
    """
    Checks for keywords in the query and returns a dynamic, friendly message
    if a keyword is found.
    """
    detected_keyword = None
    for keyword in KEYWORDS:
        if keyword.lower() in query.lower():
            detected_keyword = keyword
            break

    if detected_keyword:
        messages = KEYWORD_AWARE_MESSAGES.get(language.lower(), KEYWORD_AWARE_MESSAGES["english"])
        message_template = random.choice(messages)
        return message_template.format(keyword=detected_keyword)
    
    return None

def replace_spaces_in_image_urls(response):
    # Regex pattern to find image URLs
    pattern = r'!\[Image\]\((.*?)\)!'

    # Function to replace spaces with %20
    def replace_spaces(match):
        url = match.group(1)
        return f'![Image]({url.replace(" ", "%20")})'

    # Replace spaces in image URLs
    updated_response = re.sub(pattern, replace_spaces, response)
    updated_response.replace('\\\\', '\\')
    return updated_response