# using DiffRules

# DiffRules.@define_diffrule CartesianGrids.(::DDF{ddftype,OVERDX})(x,y) = 
#     :((GradDDF(1,ddftype=ddftype,dx=1.0/OVERDX))($x,$y)) , :((GradDDF(2,ddftype=ddftype,dx=1.0/OVERDX))($x,$y))

# eval(ForwardDiff.binary_dual_definition(:CartesianGrids, :(::DDF{ddftype,OVERDX})))