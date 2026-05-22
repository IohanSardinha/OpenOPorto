from ..IMob.ActivityTypes import IMobActivity

def PlaceCategoryMapper(cat, person):
    if cat == IMobActivity.WORK:
        if person[5] == "Worker 1 sec":
            return ["workplace_1st_sec"]
        elif person[5] == "Worker 2 sec":
            return ["workplace_2nd_sec"]
        elif person[5] == "Worker 3 sec":
            return ["workplace_3rd_sec"]
    
    elif cat == IMobActivity.TAKE_SOMEONE_SOMEWHERE:
        pass
    
    elif cat == IMobActivity.GROCERIES:
        return ["groceries","shop"]
    
    elif cat == IMobActivity.SCHOOL:
        if person[3] in ["1 Basic", "None"]:
            return ["primary_school"]
        elif person[3] in ["2 Basic", "3 Basic"]:
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