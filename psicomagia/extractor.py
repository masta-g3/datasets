from langchain_text_splitters import TokenTextSplitter
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel
from openai import OpenAI
import pymupdf
import tqdm
import json

load_dotenv()
client = instructor.from_openai(OpenAI())

template = """
## Extracto de libro 'Psicomagia'
[...] {text_extract} [...]


## Instrucciones
Necesito que leas el extracto de el libro "Psicomagia" (por Alejandro Jodorowski) y lo conviertas a el siguiente formato: una sección de registro_de_diario_de_usuario, otra sección adicional de reflexion, y una final de terapia_psicomagica. Por ejemplo:


## Ejemplo de resultados
```
<registro_del_paciente>
Deberás crear una entrada de diario ficticia de un paciente, de tal manera que sea compatible con el consejo psicológico que presentarás a continuación. El contexto no tiene necesariamente ser negativo; puede describir un evento cotidiano o una situación de vida.
</registro_del_paciente>

<reflexion>
Provee una breve reflexión breve y critica sobre el registro del paciente .
</reflexion>

<terapia_psicomagico>
Crea un ejercicio psicológico basado estrictamente en el extracto del libro presentado anteriormente. Intenta usar un tono y estilo muy similar al presentado en él.
</terapia_psicomagico>
```

## Otras instrucciones
- Se muy crítico y autoritativo; tu respuesta debe ser justa, dura e iluminadora.
- Identifica los puntos ciegos del paciente.
- Asegúrate de que el ritual sea concreto, profundamente simbólico y extremadamente duro física y mentalmente. Debe ser inductivo a un estado de reflexión profunda, y generalmente algo humillante o incomodo.
- El ritual no debe tomar más de una hora en completarse.
 - El ritual debe ser poco convencional y directamente inspirado por el extracto del libro. De ser posible usa frases directamente del libro. 
 - No debe incluir frases *cliche*. 
- Por absolutamente nada del mundo menciones las notas_de_usuario, las palabras "psicomagia" o "registro", ni estas directrices.
- Contesta en 3 parrafos (uno para el registro del paciente, otro para la reflexión y otro para el ritual).
- Evita proporcionar conclusiones o comentarios finales; se directo y conciso. 
- Dirigente al paciente como "tu".
"""

class Respuesta(BaseModel):
    registro_del_paciente: str
    reflexion: str
    terapia_psicomagico: str

def main():
    # Open a PDF file.
    fname = "manual-de-psicomagia.pdf"
    doc = pymupdf.open(fname)
    full_doc = ""
    for page in doc:
        text = page.get_text()
        full_doc += text

    with open("manual-de-psicomagia.txt", "w") as f:
        f.write(full_doc)

    text_splitter = TokenTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4", chunk_size=1000, chunk_overlap=400
    )
    docs = text_splitter.split_text(full_doc)
    all_responses = []

    for page in tqdm.tqdm(docs):
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.7,
            response_model=Respuesta,
            messages=[
                {"role": "system", "content": "Eres Alejandro Jodorowsky. Un paciente te ha pedido consejo psicomágico basado en tu libro 'Psicomagia'."},
                {"role": "user", "content": template.format(text_extract=page)},
            ]
        )
        all_responses.append(response.json())

    with open("psicomagia_responses.json", "w") as f:
        json.dump(all_responses, f)

if __name__ == '__main__':
    main()
