%
% (c) Salvatore Ruggieri, 2012-2013
% "http://www.di.unipi.it/~ruggieri"
%
% Learning from Polyhedral Sets version 1.0 - August 2013
% 

:- set_prolog_flag(toplevel_print_options, [quoted(true), portray(true)]). % to print full list

:- use_module(library(lists)).
:- use_module(library(clpr)).
% use rationals instead of reals
% :- use_module(library(clpq)). % make_unit must be disabled (see its definition)

%%%%%%% learning procedure
% S. Ruggieri. Learning from Polyhedral Sets. IJCAI 2013
%
% Ex. from Introduction:- Ps = [[0 =< X, X =< 3, 0 =< Y, Y =< 2], [0 =< X, X =< 2, 0 =< Y, Y =< 3]], learning(Ps, Cone, Ms, CHull, Base, Learned).
% Ex. 3.1 (inverse, actually)
% :- Ps = [[X =< 1, Y =< 1, X+Y =< 1], [X =< 1, Y =< 1, X+Y =< 2]], learning(Ps, Cone, Ms, CHull, Base, Learned).
% Ex. 4.2
% :- Ps = [[X =< 0, Y =< 0], [X =< 2, Y =< 0]], learning(Ps, Cone, Ms, CHull, Base, Learned).
% Ex. 4.3
% :- Ps1 = [X =< 0, Y =< 0], Cone = [X =< B, Y =< B, Y =< 2-B, 0 =< B, -2 =< -B], dec_eq(Ps1, Cone).
% :- Ps2 = [X =< 2, Y =< 0], Cone = [X =< B, Y =< B, Y =< 2-B, 0 =< B, -2 =< -B], dec_eq(Ps2, Cone).
% Ex. 4.4
% :- Ps1 = [X =< 3, Y =< 3, X+Y =< 3], R1 = [X =< A, Y =< A, X+Y =< 3, 1 =< A, -3 =< -A], dec_eq(Ps1, R1).
% :- Ps2 = [X =< 1, Y =< 1], R1 = [X =< A, Y =< A, X+Y =< 3, 1 =< A, -3 =< -A], dec_eq(Ps2, R1).
% :- Ps1 = [X =< 3, Y =< 3, X+Y =< 3], R2 = [X =< A, Y =< A, X+Y =< 3, 1 =< A, -3 =< -A, X+2*Y =< 1.5*A + 1.5], dec_eq(Ps1, R2).
% :- Ps2 = [X =< 1, Y =< 1], R2 = [X =< A, Y =< A, X+Y =< 3, 1 =< A, -3 =< -A, X+2*Y =< 1.5*A + 1.5], dec_eq(Ps2, R2).
% :- Ps3 = [X =< 2, Y =< 2, X+Y =< 3], R1 = [X =< A, Y =< A, X+Y =< 3, 1 =< A, -3 =< -A], dec_eq(Ps3, R1).
% :- Ps3 = [X =< 2, Y =< 2, X+Y =< 3], R2 = [X =< A, Y =< A, X+Y =< 3, 1 =< A, -3 =< -A, -3 =< -A, X+2*Y =< 1.5*A + 1.5], dec_eq(Ps3, R2).
% Ex. 4.5
% :- P=[-Y =< 0, X+Y =< 3, X =< 2],  homo(P,H), generate_par_cone([H], ParCone), par_cone_maxima([P], ParCone, [M]).
% Ex. 4.6
% :- Ps=[[-Y =< 0, X+Y =< 3, X =< 2],[-Y =< 0, X+Y =< 1], [-Y =< 0, X+Y =< 2]], learning(Ps, Cone, Ms, CHull, Base, _).
% Ex. 4.7
%:- Ps=[[X =< 0], [Y =< 0]], learning(Ps, Cone, Ms, CHull, Base, Learned).
% Ex. 4.9 and 4.10
% :-  C
%        the answered Learned system does not include "X =< B" because it is a redundant inequality (implied by "0 =< Y, X+Y =< B")
% Ex. 4.11
% :- Ps=[[X =< 0, Y =< 0],[X =< 1, Y =< 2], [X =< 2, Y =< 1]], learning(Ps, Cone, Ms, CHull, Base, Learned).
%
%%%%%%%

% fuck you prolog
:- use_module(library(http/thread_httpd)).
:- use_module(library(http/http_dispatch)).
:- use_module(library(pengines)).

server(Port) :- http_server(http_dispatch, [port(Port)]).

:- server(4242).

%Ps = [[X =< 0, Y =< 0], [X =< 2, Y =< 0]], learning(Ps, Cone, Ms, CHull, Base, Learned).
% start swipl and copy Ps=...
learning(Ps, Learned) :-
	learning(Ps, _, _, _, _, Learned).

learning(Ps, Cone, Ms, CHull, Base, Learned) :-
	standardizeAll(Ps, StdPs),
	same_cone(StdPs, HomoPs),
	(var(Cone) -> generate_par_cone(HomoPs, ParCone);true),
	par_cone_maxima(StdPs, ParCone, Ms),
	convex_hull(Ms, CHull),
	append(ParCone, CHull, Base),
	term_variables(HomoPs, Vars),
	term_variables(Ms, Pars),
	lps_gauss(Base, Pars, Vars, Pars1, Base1),
	lps_project(Ms, StdPs, Pars1, Vars, Base1, Learned).

%:- learning([[X =< 1, Y =< 1, X+Y =< 1], [X =< 1, Y =< 1, X+Y =< 2]], Cone, Ms, CHull, Base, Learned).

%%%% Gauss projection - it does not need to check that projection is a generalization

lps_gauss(Base, [], _, [], Base).
lps_gauss(Base, [P|Ps], Rest, Pars1, Learned) :-
	(is_constant(P, Base, _) ->
		append(Ps, Rest, Pars),
		project(Pars, Base, Base1),
		lps_gauss(Base1, Ps, Rest, Pars1, Learned)
	;
		Pars1 = [P|Pars1rec],
		lps_gauss(Base, Ps, [P|Rest], Pars1rec, Learned)
	).
	
%%%% General projection - it does need to check that projection is a generalization

lps_project(_, _, [], _, Base, Base).
lps_project(Ms, Polys, [P|Ps], Rest, Base, Learned) :-
	append(Rest, Ps, Pars),
	project(Pars, Base, BaseProj),
	(is_a_generalization(BaseProj, Polys, Ms) ->
		lps_gauss(BaseProj, Ps, Pars, Ps1, Base1),
		lps_project(Ms, Polys, Ps1, Rest, Base1, Learned)
	;
		lps_project(Ms, Polys, Ps, [P|Rest], Base, Learned)).

%%%% check that a parameterized system Base is a generalization of given systems Ps for given parameter values Ms

is_a_generalization(_, [], []).
is_a_generalization(Base, [P|Ps], [M|Ms]) :-
	copy_term(Base-P-M, CopyBase-CopyP-CopyM),
	tell_cs(CopyM),
	equivalent(CopyBase, CopyP),
	is_a_generalization(Base, Ps, Ms).

%%%% compute the maxima of the ParCone wrt the systems Ps

par_cone_maxima([], _, []).
par_cone_maxima([P|Ps], ParCone, [M|Ms]) :-
	dec_inc(P, ParCone, Leq),
	leq2eq(Leq, M),
	par_cone_maxima(Ps, ParCone, Ms).

leq2eq([], []).
leq2eq([E1 =< E2|Xs], [E1 = E2|Ys]) :-
	leq2eq(Xs, Ys).
leq2eq([E1 = E2|Xs], [E1 = E2|Ys]) :-
	leq2eq(Xs, Ys).

%%%% generate cone of the base system
% :- same_cone([ [X =< 3], [2*X =< 5]], H), generate_par_cone(H, Cone).
%%%%

generate_par_cone(H, ParCone) :-
	generate_cone_nopar(H, ConeNoPar),
	generate_cone_addpar(ConeNoPar, ParCone).

generate_cone_addpar([], []).
generate_cone_addpar([E =< 0|Hs], [E =< _|Cs]) :-
	generate_cone_addpar(Hs, Cs).

generate_cone_nopar([], []).
generate_cone_nopar([C|Cs], Cone) :-
	generate_cone_nopar(Cs, Cone1),
	list_to_ord_set(C, Cord),  % PLANNED: X + Y  is treated as different than Y + X
	ord_union(Cone1, Cord, Cone).

%%%% check that list of systems have same cone, and build list of cones
% :- same_cone([ [X =< 3], [2*X =< 5]], H).
%%%%

same_cone([], []).
same_cone([C], [H]) :-
	homo_unit(C, H).
same_cone([C,C1|Cs], [H|Hs]) :-
	homo_unit(C, H),
	same_cone(H, [C1|Cs], Hs).

same_cone(_, [], []).
same_cone(H, [C|Cs], [H1|Hs]) :-
	homo_unit(C, H1),
	equivalent(H, H1),
	same_cone(H, Cs, Hs).


%%%%%%% additions to ECAI 2012 paper
%
% :- dec_eq([0 =< X, 0 =< Y, X+Y =<1], [-X =< 0, X =< A, -Y =< 0, Y =< B, X+Y =< C], Sol).
%%%%%%%

%%% PLANNED: add a modified standardize(P, V, ParP) to move variables in V (parameters) to RHS, and all others to LHS - now the program relies on the user input

%%%% membership conditions in the third argument
dec_eq(P, ParP, Sol) :-
  	copy_term(P-ParP, CopyP-CopyParP),
  	(tell_cs(CopyP) -> % satisfiability test
		maxima(CopyParP, ParP, CxK, Leq),
		entails(CxK, P),
		minimal([], CxK, Leq, Eq),
		append(Leq, Eq, Sol),
		copy_term(Sol, CSol),
		tell_cs(CSol)
	;
		dec_eq_empty(CopyParP) -> Sol=[] % in this case, the conditions are not calculated
	).
	
	
%%%%%%% membership of a polyhedron to the class of a parameterized polyhedron - code adapted from
% S. Ruggieri. Deciding Membership in a Class of Polyhedra. 20th European Conference on Artificial Intelligence (ECAI 2012): 702-707. IOS Press, 2012
%
% :- dec_eq([0 =< X1, X1 =< 1, 0 =< X2, X2 =< 1], [0 =< X1, 0 =< X2, X1 =< 1 + R1, X1 =< 2-R2, X1 =< 1+R3, X2 =< 2-R1, X2 =< 1+R2]).
% :- dec_eq([0 =< X, 0 =< Y, X+Y =<1], [-X =< 0, X =< A, -Y =< 0, Y =< B, X+Y =< C]).
% :- dec_eq([1 =< 0], [X+Y =< A]).
%%%%%%%

%%%% membership conditions in the constraint store
dec_eq(P, ParP) :-
  	copy_term(P-ParP, CopyP-CopyParP),
  	(tell_cs(CopyP) -> % satisfiability test
		maxima(CopyParP, ParP, CxK, Leq),
		tell_cs(Leq),
		entails(CxK, P),
		minimal([], CxK, Leq, Eq),
		tell_cs(Eq)
	;
		dec_eq_empty(CopyParP)).

maxima([], [], [], []).
maxima([E =< _P|Cs], [E1 =< P1|C1s], [E1 =< K|CxK], [K=<P1|Leq]) :-
    sup(E, K),
    maxima(Cs, C1s, CxK, Leq).
		
minimal(Sub, [], [], []) :-
    irredundant([], Sub).
minimal(Sub, [C|Cs], [K=<P|Leq], [K=P|Eq]) :-
    \+ entails(Sub, C),
    minimal([C|Sub], Cs, Leq, Eq).
minimal(Sub, [C|Cs], [_|Leq], Eq) :-
    append(Sub, Cs, Rest),
    entails(Rest, [C]),
    minimal(Sub, Cs, Leq, Eq).

irredundant(_, []).
irredundant(Ss, [C|Cs]) :-
	append(Ss, Cs, All),
	\+ entails(All, [C]),
	irredundant([C|Ss], Cs).


%%%%%%% membership of the empty polyhedron to the class of a parameterized polyhedron - code adapted from
% --- ADD CITATION HERE ---
%
% :- dec_eq_empty([0 =< 1]).
% :- dec_eq_empty([1 =< 0]).
% :- dec_eq_empty([X =< A, -X =< B]). % differently from dec_eq, it describes a single condition on parameters
%%%%%%%

dec_eq_empty(ParP) :-
	( satisfiable(ParP) -> 
		( once(has_parameter(ParP, A)) ->
			copy_term(ParP-A, CopyParP-CopyA),
			tell_cs(CopyParP),
			( sup(CopyA, K) -> 
				A=K+1,
				true
				;
				( inf(CopyA, K) ->
					A=K-1,
					true
					;
					A=0,
					dec_eq_empty(ParP)
				)
			)
			;
			false
		)
	;
		true).

has_parameter([_ =< P|_], A) :-
	has_var(P, A).
has_parameter([_|Xs], A) :-
	has_parameter(Xs, A).
	
has_var(X, X) :- var(X).
has_var(N*X, X) :- ground(N), var(X).
has_var(X*N, X) :- ground(N), var(X).
has_var(-X, A) :- has_var(X, A).
has_var(X+_, A) :- has_var(X, A).
has_var(_+Y, A) :- has_var(Y, A).
has_var(X-_, A) :- has_var(X, A).
has_var(_-Y, A) :- has_var(Y, A).


%%%%%%% inclusion of a polyhedron in the class of a parameterized polyhedron - code adapted from
% S. Ruggieri. Deciding Membership in a Class of Polyhedra. 20th European Conference on Artificial Intelligence (ECAI 2012): 702-707. IOS Press, 2012
%
% variables in the RHS of a parameterized polyhedra are parameters
% :- dec_inc([0 =< X, 0 =< Y, X+Y =<1], [-X =< 0, X =< A, -Y =< 0, Y =< B, X+Y =< C], Leq).
% :- dec_inc([0 =< X, 0 =< Y, X+Y =<1], [-X =< 0, X =< A, -Y =< 0, Y =< B, X+Y =< C]).
%%%%%%%
	
%%%% inclusion conditions in the constraint store
dec_inc(P, ParP) :-
	dec_inc(P, ParP, Leq),
	tell_cs(Leq).

%%%% inclusion conditions in the third argument
dec_inc(P, ParP, Leq) :-
	copy_term(P-ParP, CopyP-CopyParP),
  	(tell_cs(CopyP) -> % satisfiability test
		maxima(CopyParP, ParP, Leq)
	;
		Leq=[]).

maxima([], [], []).
maxima([E =< _P|Cs], [_E1 =< P1|C1s], Leq1) :-
    sup(E, K), 
	(K == P1 -> % remove trivially true inequalities
		Leq1 = Leq 
	;
		Leq1 = [K=<P1|Leq]),
    maxima(Cs, C1s, Leq).

	
%%%%%%% basic operations on linear systems
%  satisfiable, entails, equivalent
%
% :- equivalent([X =< Y+3],[-3*Y =< -3*X+9]).
%%%%%%%%%%%%%%

equivalent(S, C) :-
	entails(S, C),
	entails(C, S).

entails(S, C) :-
	copy_term(S-C, S1-C1),
	tell_cs(S1),
	is_entailed(C1).

is_entailed([]).
is_entailed([C|Cs]) :- 
	entailed(C),
	is_entailed(Cs).

satisfiable(P) :-
	copy_term(P, CopyP),
	tell_cs(CopyP).	

is_constant(V, P, C) :-
	copy_term(V-P, CopyV-CopyP),
	tell_cs(CopyP),
	sup(CopyV, C), 
	inf(CopyV, C).

tell_cs([]).
tell_cs([C|Cs]) :-  
	{C}, 
	tell_cs(Cs).

%%%%%%% homogeneous and unitary versions of a linear system
% :- homo([X =< Y+3],H).
% :- homo_unit([X =< Y+3],H).
%%%%%%%

homo([], []).
homo([C|Cs],[H|Hs]) :-
	C =.. [RelOp, E1, E2],
	once(move_constants(E1-E2, EH, _)),
	H =.. [RelOp, EH, 0],
	homo(Cs, Hs).
	
homo_unit([], []).
homo_unit([C|Cs],[H|Hs]) :-
	C =.. [RelOp, E1, E2],
	once(move_constants(E1-E2, EH, _)),
	once(make_unit(EH, UH)),
	H =.. [RelOp, UH, 0],
	homo_unit(Cs, Hs).

%%% make expressions unitary 
% :- make_unit(X + 2*Y, U).
%%%

% comment next line to enable make unit
make_unit(X, X).

make_unit(E, U) :-
	collect_abs_coeff(E, C),
	l2normsq(C, L2sq),
	L2 is sqrt(L2sq),
	divide_coeff(E, L2, U).
	
l2normsq([N], M) :- 
	M is N*N.

l2normsq([N1, N2|Ns], M) :- 
	l2normsq([N2|Ns], M2),
	M is N1*N1 + M2.

collect_abs_coeff(X, [1]) :- var(X).
collect_abs_coeff(N*X, [N]) :- ground(N), var(X).
collect_abs_coeff(X*N, [N]) :- ground(N), var(X).
collect_abs_coeff(-X, C) :- collect_abs_coeff(X, C).
collect_abs_coeff(X+Y, C) :- 
	collect_abs_coeff(X, C1), 
	collect_abs_coeff(Y, C2),
	append(C1,C2,C).
collect_abs_coeff(X-Y, C) :- 
	collect_abs_coeff(X, C1), 
	collect_abs_coeff(Y, C2),
	append(C1,C2,C).

divide_coeff(X, L, M*X) :- var(X), M is 1 / L.
divide_coeff(N*X, L, M*X) :- ground(N), var(X), M is N / L.
divide_coeff(X*N, L, M*X) :- ground(N), var(X), M is N / L.
divide_coeff(-X, L, X1) :- divide_coeff(X, -L, X1).
divide_coeff(X+Y, L, X1+Y1) :- 
	divide_coeff(X, L, X1), 
	divide_coeff(Y, L, Y1).
divide_coeff(X-Y, L, X1-Y1) :- 
	divide_coeff(X, L, X1), 
	divide_coeff(Y, L, Y1).

%%%%%%% move constant terms to RHS and variables to LHS of a linear system
% :- standardize([X =< Y+3],H).
% :- standardize([X = Y+3],H).
%%%%%%%

standardizeAll([], []).
standardizeAll([C|Cs], [SC|SCs]) :-
	standardize(C, SC),
	standardizeAll(Cs, SCs).

standardize([], []).
standardize([E1 = E2|Cs], Hs) :-
	standardize([E1 =< E2, E2 =< E1 |Cs], Hs).
standardize([E1 >= E2|Cs],Hs) :-
	standardize([E2 =< E1 |Cs], Hs).

standardize([E1 =< E2|Cs],[EH =< A|Hs]) :-
	once(move_constants(E1, EH1, A1)),
	once(move_constants(E2, EH2, A2)),
	A is A2-A1,
	comp_exp(EH1, EH2, EH),
	standardize(Cs, Hs).

comp_exp(0, X, -X) :- var(X), !.
comp_exp(E1, X, E1-X) :- var(X), !.
comp_exp(E1, 0, E1).
comp_exp(0, N*X, -N*X) :- ground(N), var(X), !.
comp_exp(E1, N*X, E1-N*X) :- ground(N), var(X).
comp_exp(0, -X, X) :- !.
comp_exp(E1, -X, E1+X).
comp_exp(E1, X+Y, E) :- comp_exp(E1, X, E2), comp_exp(E2,Y,E).
comp_exp(E1, X-Y, Y) :- comp_exp(E1, X, 0), !.
comp_exp(E1, X-Y, E2+Y) :- comp_exp(E1, X, E2).
	
move_constants(N, 0, N) :- ground(N).
move_constants(X, X, 0) :- var(X).
move_constants(N*X, N*X, 0) :- ground(N), var(X).
move_constants(X*N, N*X, 0) :- ground(N), var(X).
move_constants(X+Y, R, A) :- 
	move_constants(X, X1, A1), 
	move_constants(Y, Y1, B1),
	A is A1+B1,
   (Y1 == 0 -> R = X1; (X1 == 0 -> R = Y1; R = X1+Y1)).
move_constants(X-Y, R, A) :- 
	move_constants(X, X1, A1), 
	move_constants(Y, Y1, B1),
	A is A1+B1,
   (Y1 == 0 -> R = X1; (X1 == 0 -> R = -Y1; R = X1-Y1)).
move_constants(-X, -X1, -A) :- move_constants(X, X1, A).
	

%%%%%%% additions to convex hull code

%%%% convex hull of a list of polyhedra
% :- convex_hull([[X=0,Y=0],[X=1,Y=1],[X=2,Y=0]],S).
%%%%%%%

convex_hull([], []).
convex_hull([Cs], Cs).
convex_hull([C1s, C2s|CLs], Czs) :-
	convex_hull([C2s|CLs], CH1),
	convex_hull(C1s, CH1, Czs).

%%%% convex hull of two polyhedra
% :- convex_hull([X=0,Y=1],[X>=0,Y=X],S).
%%%%%%%

convex_hull(Cxs, Cys, Czs) :-
	term_variables(Cxs-Cys, Xs),
	copy_term(Xs-Cys, Ys-CCys),
	convex_hull(Xs, Cxs, Ys, CCys, Zs, Czs),
	Xs=Zs.

	
%%%%%%% convex hull code from
% F. Benoy, A. King, F. Mesnard: Computing convex hulls with a linear solver. TPLP 5(1-2): 259-271 (2005)
%
% :- convex_hull([X1,Y1],[X1=0,Y1=1],[X2,Y2],[X2>=0,Y2=X2],V,S), X1=X2, Y1=Y2, V = [X1,Y1].
%%%%%%%

convex_hull(Xs, Cxs, Ys, Cys, Zs, Czs) :-
	scale(Cxs, Sig1, [], C1s),
	scale(Cys, Sig2, C1s, C2s),
	add_vect(Xs, Ys, Zs, C2s, C3s),
	project(Zs, [Sig1 >= 0, Sig2 >= 0, Sig1+Sig2 = 1|C3s], Czs).

scale([], _, Cs, Cs).
scale([C1|C1s], Sig, C2s, C3s) :-
	C1 =.. [RelOp, A1, B1],
	C2 =.. [RelOp, A2, B2],
	mul_exp(A1, Sig, A2),
	mul_exp(B1, Sig, B2),
	scale(C1s, Sig, [C2|C2s], C3s).
	
mul_exp(E1, Sigma, E2) :- once(mulexp(E1, Sigma, E2)).

mulexp( X, _, X) :- var(X).
mulexp(N*X, _, N*X) :- ground(N), var(X).
mulexp(X*N, _, N*X) :- ground(N), var(X). % added to JLP
mulexp( -X, Sig, -Y) :- mulexp(X, Sig, Y).
mulexp(A+B, Sig, C+D) :- mulexp(A, Sig, C), mulexp(B, Sig, D).
mulexp(A-B, Sig, C-D) :- mulexp(A, Sig, C), mulexp(B, Sig, D).
mulexp( N, Sig, N*Sig) :- ground(N).

add_vect([], [], [], Cs, Cs).
add_vect([U|Us], [V|Vs], [W|Ws], C1s, C2s) :-
	add_vect(Us, Vs, Ws, [W = U+V|C1s], C2s).

project(Xs, Cxs, ProjectCxs) :-
%	call_residue(						%%% call_residue not available in SWI prolog
	copy_term(Xs-Cxs, CpyXs-CpyCxs),
%	_),
	tell_cs(CpyCxs),
	prepare_dump(CpyXs, Xs, Zs, DumpCxs, ProjectCxs),
	dump(Zs, Vs, DumpCxs), Xs = Vs.

prepare_dump([], [], [], Cs, Cs).
prepare_dump([X|Xs], YsIn, ZsOut, CsIn, CsOut) :-
	(ground(X) ->
	YsIn = [Y|Ys],
	ZsOut = [_|Zs],
	CsOut = [Y=X|Cs]
	;
	YsIn = [_|Ys],
	ZsOut = [X|Zs],
	CsOut = Cs
	),
	prepare_dump(Xs, Ys, Zs, CsIn, Cs).

% duplicate predicate definition - commented out
% 
%tell_cs([]).
%tell_cs([C|Cs]) :-  
%	{C}, 
%	tell_cs(Cs).
