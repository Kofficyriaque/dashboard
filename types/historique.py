from pydantic import BaseModel


class historique (BaseModel):
    salaire_predit: int
    salaire_min : int
    salaire_mensuel: int  
    niveau_experience: str
    description: str
    competences: str
    region: str
    titre: str