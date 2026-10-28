import json
import numpy as np
import pytest
from anyloads import (AiryEnvironment, AiryComponent, CurrentProfile, HydrodynamicOwnership,
    MorisonSection, morison_force, environment_to_dict, environment_from_dict, LoadError)

def test_airy_dispersion_profiles_and_eulerian_time_derivative():
    e=AiryEnvironment(10.,(AiryComponent(.2,1.1,30.,.4),),CurrentProfile.uniform(10.,(2.,0.,0.)))
    k=e.wave_numbers[0]
    assert abs(9.81*k*np.tanh(10*k)-1.1**2)<1e-12
    point=np.array([1.,2.,-3.]);t=.6;eps=1e-5
    f=e.evaluate(point,t)
    direction=np.array([np.cos(np.pi/6),np.sin(np.pi/6),0.])
    phase=k*point@direction-1.1*t+.4
    expected=.2*1.1*np.cosh(k*7)/np.sinh(k*10)*np.cos(phase)*direction
    expected[2]=.2*1.1*np.sinh(k*7)/np.sinh(k*10)*np.sin(phase)
    np.testing.assert_allclose(f.wave_velocity,expected,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose((e.evaluate(point,t+eps).velocity-e.evaluate(point,t-eps).velocity)/(2*eps),
                               f.acceleration,rtol=1e-8,atol=1e-10)
    for j in range(3):
        d=np.eye(3)[j]*eps
        np.testing.assert_allclose((e.evaluate(point+d,t).velocity-e.evaluate(point-d,t).velocity)/(2*eps),
                                  f.velocity_gradient[:,j],rtol=1e-8,atol=1e-10)
        np.testing.assert_allclose((e.evaluate(point+d,t).acceleration-e.evaluate(point-d,t).acceleration)/(2*eps),
                                  f.acceleration_gradient[:,j],rtol=1e-8,atol=1e-10)
    assert e.evaluate([0.,0.,-10.],t).wave_velocity[2]==0.

def test_current_batch_gradient_and_no_steady_acceleration():
    e=AiryEnvironment(10.,current=CurrentProfile((0.,5.,10.),((2.,0.,0.),(1.,1.,0.),(0.,1.,0.))))
    f=e.evaluate([[0.,0.,-2.],[1.,0.,-7.]],0.)
    np.testing.assert_allclose(f.velocity,[[1.6,.4,0.],[.6,1.,0.]],atol=1e-14)
    np.testing.assert_array_equal(f.acceleration,np.zeros((2,3)))
    np.testing.assert_allclose(f.velocity_gradient[:,:,2],[[.2,-.2,0.],[.2,0.,0.]])

def test_seeded_spectrum_roundtrip_keeps_realization_exact():
    e=AiryEnvironment.seeded_spectrum(20.,[.7,1.,1.4],[.2,.4,.1],[.1,.1,.1],seed=17)
    restored=environment_from_dict(json.loads(json.dumps(environment_to_dict(e))))
    points=np.array([[1.,2.,-3.],[3.,1.,-9.]])
    for time in (0.,.3,2.):
        np.testing.assert_array_equal(restored.evaluate(points,time).velocity,e.evaluate(points,time).velocity)
    other=AiryEnvironment.seeded_spectrum(20.,[.7,1.,1.4],[.2,.4,.1],[.1,.1,.1],seed=18)
    assert e.components!=other.components

@pytest.mark.parametrize('changes',({'acceleration_convention':'material'}, {'current':False},
    {'water_depth':True}, {'components':(AiryComponent(.1,1e300),)}))
def test_environment_refuses_unsupported_or_invalid_inputs(changes):
    values=dict(water_depth=10.)|changes
    with pytest.raises(LoadError):
        AiryEnvironment(**values)

@pytest.mark.parametrize('values',(
    dict(primary_role='morison_full',bem_included=True),
    dict(primary_role='morison_full',drag_correction=True),
    dict(primary_role='potential_flow',bem_included=True,buoyancy_owner='morison'),
    dict(primary_role='none',buoyancy_owner='potential_flow'),
))
def test_duplicate_ownership_is_refused(values):
    with pytest.raises(LoadError):
        HydrodynamicOwnership(**values)

def test_vectorized_morison_law_derivatives_diagnostics_and_relative_dissipation():
    section=MorisonSection(.4,.7,1.2);role=HydrodynamicOwnership(buoyancy_owner='morison')
    u=np.array([[2.,.5,0.],[1.,-.2,.3]])
    af=np.array([[.2,-.1,.4],[.1,.3,-.2]])
    v=np.array([[1.,.2,.1],[-.1,.4,.1]])
    a=np.array([[.3,.2,.1],[.2,.1,.4]])
    t=np.array([0.,0.,1.])
    args=[u,af,v,a,t]
    f=morison_force(section,role,1025.,*args,wave_velocity=u-.5,current_velocity=np.zeros_like(u)+.5)
    P=np.diag([1.,1.,0.]);w=(u-v)@P
    expected=1025*section.area*(af@P)+1025*1.2*section.area*((af-a)@P)+.5*1025*.7*.4*np.linalg.norm(w,axis=1)[:,None]*w
    expected[:,2]+=1025*9.81*section.area
    np.testing.assert_allclose(f.force,expected,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(sum(f.components.values()),f.force,atol=1e-12)
    np.testing.assert_allclose(sum(f.drag_attribution.values()),f.components['drag'],atol=1e-12)
    assert np.all(f.relative_drag_power<=1e-12)
    derivatives=[f.fluid_velocity_derivative,f.fluid_acceleration_derivative,
                 f.structure_velocity_derivative,f.structure_acceleration_derivative]
    for arg_index,derivative in enumerate(derivatives):
        for j in range(3):
            plus=[x.copy() for x in args];minus=[x.copy() for x in args]
            plus[arg_index][:,j]+=1e-5;minus[arg_index][:,j]-=1e-5
            measured=(morison_force(section,role,1025.,*plus).force-morison_force(section,role,1025.,*minus).force)/2e-5
            np.testing.assert_allclose(measured,derivative[:,:,j],rtol=1e-8,atol=1e-8)
    for j in (0,1):
        plus=args[:-1]+[t+np.eye(3)[j]*1e-5];minus=args[:-1]+[t-np.eye(3)[j]*1e-5]
        plus[-1]/=np.linalg.norm(plus[-1]);minus[-1]/=np.linalg.norm(minus[-1])
        measured=(morison_force(section,role,1025.,*plus).force-morison_force(section,role,1025.,*minus).force)/2e-5
        np.testing.assert_allclose(measured,f.axis_derivative[:,:,j],rtol=1e-8,atol=1e-7)

def test_drag_only_has_no_inertia_buoyancy_and_zero_relative_speed_is_smooth():
    section=MorisonSection(.4)
    role=HydrodynamicOwnership('potential_flow',True,True,'potential_flow')
    f=morison_force(section,role,1025.,[1.,0.,0.],[2.,3.,4.],[1.,0.,0.],[5.,6.,7.],[0.,0.,1.])
    np.testing.assert_array_equal(f.force,np.zeros(3))
    for derivative in (f.structure_acceleration_derivative,f.fluid_acceleration_derivative,f.structure_velocity_derivative):
        np.testing.assert_array_equal(derivative,np.zeros((3,3)))

def test_unknown_serialized_fields_never_execute_code():
    data=environment_to_dict(AiryEnvironment(10.));data['python_code']='raise RuntimeError'
    with pytest.raises(LoadError,match='fields/version'):
        environment_from_dict(data)
