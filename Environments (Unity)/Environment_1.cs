using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class AgentCube : Agent
{
    public float speed = 10f;
    public Transform Target;
    public GameObject ground;
    [HideInInspector] public Bounds areaBounds;
    [HideInInspector] public bool useVectorObs;

    void Start()
    {
        areaBounds = ground.GetComponent<Collider>().bounds;
    }

    public override void AgentReset()
    {
        // Agent: reset position
        this.transform.position = ground.transform.position + new Vector3(0, 0.5f, 0);

        // Target: new random position
        var randomPosX = Random.Range(-areaBounds.extents.x,areaBounds.extents.x);
        var randomPosZ = Random.Range(-areaBounds.extents.z, areaBounds.extents.z);
        var randomSpawnPos = ground.transform.position + new Vector3(randomPosX, 0.5f, randomPosZ);
        Target.position = randomSpawnPos;
    }

    public void MoveAgent(float[] act)
    {
        var action = Mathf.FloorToInt(act[0]);

        switch (action)
        {
            case 1:
                this.transform.position = transform.position + new Vector3(0, 0, speed * Time.deltaTime);
                break;
            case 2:
                this.transform.position = transform.position - new Vector3(0, 0, speed * Time.deltaTime);
                break;
            case 3:
                this.transform.position = transform.position + new Vector3(speed * Time.deltaTime, 0, 0);
                break;
            case 4:
                this.transform.position = transform.position - new Vector3(speed * Time.deltaTime, 0, 0);
                break;
        }
    }

    public override void AgentAction(float[] vectorAction)
    {
        // Punish slightly to encourage hurrying up
        AddReward(-1f / agentParameters.maxStep);

        // Move the agent using the action
        MoveAgent(vectorAction);

        // Fell off platform
        if (this.transform.position.y < 0.45f)
        {
            AddReward(-1f);
            AgentReset();
            Done();
        }

    }

    public override float[] Heuristic()
    {
        if (Input.GetKey(KeyCode.W))
        {
            return new float[] { 1 };
        }
        if (Input.GetKey(KeyCode.S))
        {
            return new float[] { 2 };
        }
        if (Input.GetKey(KeyCode.D))
        {
            return new float[] { 3 };
        }
        if (Input.GetKey(KeyCode.A))
        {
            return new float[] { 4 };
        }
        return new float[] { 0 };
    }

    void OnCollisionEnter(Collision col)
    {
        // Touched target
        if (col.gameObject.CompareTag("target"))
        {
            ScoredAGoal();
        }
    }

    public void ScoredAGoal()
    {
        AddReward(1f);
        AgentReset();
        Done(); 
    }
}


