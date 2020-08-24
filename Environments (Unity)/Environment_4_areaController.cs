using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AreaController : MonoBehaviour
{
    public GameObject agent1;
    public GameObject agent2;

    private void FixedUpdate()
    {
        if (GetComponentInChildren<Goal_col>().goal_on)
        {
            agent1.GetComponent<RollerAgent>().ScoredAGoal();
            agent2.GetComponent<RollerAgent>().ScoredAGoal();
            GetComponentInChildren<Goal_col>().goal_on = false;
        }
    }
}
