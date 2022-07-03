def reconstructure(constraints):
    eqpair, neqpair = [], []
    for constraint in constraints:
        (neqpair, eqpair)[constraint[1] == "="].append(constraint)

    if len(eqpair) != 0:
        equality = [constraint[0] for constraint in eqpair]
    else:
        equality = None
        
    if len(neqpair) != 0:
        neqmark = []
        neqcst = []
        for neq in neqpair:
            if neq[1] == ">" or neq[1] == ">=":
                neqmark.append(1)
            else:
                neqmark.append(0)
            neqcst.append(neq[0])
        inequality = zip(neqcst, neqmark)
    else:
        inequality = None
        
    return equality, list(inequality)
