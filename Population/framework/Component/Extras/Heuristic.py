class IMobActivity:
    WORK = "work"
    TAKE_SOMEONE_SOMEWHERE = "take_someone_somewhere"
    HOME = "home"
    GROCERIES = "groceries"
    SCHOOL = "school"
    AROUND_THE_BLOCK = "around_the_block"
    WORKOUT = "workout"
    VISIT_FRIEND_FAMILY = "visit_friend_family"
    EAT_OUT = "eat_out"
    OTHER = "other"
    LEASURE_SPORT_OR_CULURAL = "leasure_sport_or_culural"
    PERSONAL_ISSUES = "personal_issues"
    LEASURE_OTHER = "leasure_other"
    DOCTOR = "doctor"
    LEASURE_COLLECTIVE = "leasure_collective"

def _get_person_field(person, field_name, fallback_index):
    if isinstance(person, dict):
        return person[field_name]

    if hasattr(person, field_name):
        return getattr(person, field_name)

    return person[fallback_index]


def PlaceCategoryMapper(cat, person):
    if cat == IMobActivity.WORK:
        economic_situation = _get_person_field(person, "economicSituation", 5)

        if economic_situation == "Worker 1 sec":
            return ["workplace_1st_sec"]
        elif economic_situation == "Worker 2 sec":
            return ["workplace_2nd_sec"]
        elif economic_situation == "Worker 3 sec":
            return ["workplace_3rd_sec"]
    
    elif cat == IMobActivity.TAKE_SOMEONE_SOMEWHERE:
        pass
    
    elif cat == IMobActivity.GROCERIES:
        return ["groceries","shop"]
    
    elif cat == IMobActivity.SCHOOL:
        education_level = _get_person_field(person, "educationLvl", 3)

        if education_level in ["1 Basic", "None"]:
            return ["primary_school"]
        elif education_level in ["2 Basic", "3 Basic"]:
            return ["secondary_school"]
        else:
            return ["university"]
    
    elif cat == IMobActivity.AROUND_THE_BLOCK:
        pass
    elif cat == IMobActivity.WORKOUT:
        pass
    elif cat == IMobActivity.VISIT_FRIEND_FAMILY:
        pass
    elif cat == IMobActivity.EAT_OUT:
        pass
    elif cat == IMobActivity.OTHER:
        pass
    elif cat == IMobActivity.LEASURE_SPORT_OR_CULURAL:
        pass
    elif cat == IMobActivity.PERSONAL_ISSUES:
        pass
    elif cat == IMobActivity.LEASURE_OTHER:
        pass
        #return ["leisure"]
    elif cat == IMobActivity.DOCTOR:
        pass
    elif cat == IMobActivity.LEASURE_COLLECTIVE:
        pass
    
    return "ALL"