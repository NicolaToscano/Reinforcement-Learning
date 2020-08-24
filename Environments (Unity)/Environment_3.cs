using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class AgentCube : Agent
{
    public float speed = 2f;
    public float rotationSpeed = 300f;
    Rigidbody rBody;
    public GameObject object_ball;
    public GameObject ground;
    [HideInInspector] public Bounds areaBounds;
    [HideInInspector] public bool useVectorObs;

    void Start()
    {
        rBody = GetComponent<Rigidbody>();
        areaBounds = ground.GetComponent<Collider>().bounds;
    }

    public override void AgentReset()
    {
        // Agent: new random position
        this.rBody.velocity = Vector3.zero;
        this.rBody.angularVelocity = Vector3.zero;
        this.transform.position = randomPosition();

        // Ball: new random position
        object_ball.GetComponent<ball>().resetBall();
    }

    public void MoveAgent(float[] act)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;
        var action = Mathf.FloorToInt(act[0]);

        switch (action)
        {
            case 1:
                dirToGo = transform.forward * 1f;
                break;
            case 2:
                dirToGo = transform.forward * -1f;
                break;
            case 3:
                rotateDir = transform.up * 1f;
                break;
            case 4:
                rotateDir = transform.up * -1f;
                break;
        }

        transform.Rotate(rotateDir, Time.fixedDeltaTime * rotationSpeed);
        rBody.AddForce(dirToGo * speed, ForceMode.VelocityChange);
    }


    public override void AgentAction(float[] vectorAction)
    {
        // Punish slightly to encourage hurrying up
        AddReward(-1f / agentParameters.maxStep);

        // Move the agent using the action
        MoveAgent(vectorAction);

        // Goal
        if (object_ball.GetComponent<ball>().goal)
        {
            object_ball.GetComponent<ball>().resetBall();
            ScoredAGoal();
        }

    }

    public override float[] Heuristic()
    {
        if (Input.GetKey(KeyCode.D))
        {
            return new float[] { 3 };
        }
        if (Input.GetKey(KeyCode.W))
        {
            return new float[] { 1 };
        }
        if (Input.GetKey(KeyCode.A))
        {
            return new float[] { 4 };
        }
        if (Input.GetKey(KeyCode.S))
        {
            return new float[] { 2 };
        }
        return new float[] { 0 };
    }

    public void ScoredAGoal()
    {
        AddReward(1f);
        AgentReset();
        Done();
    }

    public Vector3 randomPosition()
    {
        return ground.transform.position + new Vector3(1f, 0f, Random.Range(-5f, 5f));
    }
}


