using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class AgentCube : Agent
{
    public float speed = 2f;
    public float rotationSpeed = 300f;
    private float x_agent;
    private float z_agent;
    private float x_orange;
    private float z_orange;
    private float x_red;
    private float z_red;
    public int level = 1;
    private bool orange_set = false;
    private bool red_set = false;
    Rigidbody rBody;
    public Transform Wall;
    public GameObject Block_orange;
    public GameObject Block_red;
    public GameObject ground;
    [HideInInspector] public Bounds areaBounds;
    [HideInInspector] public bool useVectorObs;

    void Start()
    {
        rBody = GetComponent<Rigidbody>();
        areaBounds = ground.GetComponent<Collider>().bounds;
        Difficulty();
    }

    public override void AgentReset()
    {
        Difficulty();

        // Agent: new random position
        this.transform.position = ground.transform.position + new Vector3(x_agent, 0.5f, z_agent);
        
        // Blocks: new random position
        Block_orange.transform.rotation = Quaternion.Euler(0f, 0f, 0f);
        Block_red.transform.rotation = Quaternion.Euler(0f, 0f, 0f);
        Block_orange.GetComponent<Rigidbody>().velocity = new Vector3(0f, 0f, 0f);
        Block_red.GetComponent<Rigidbody>().velocity = new Vector3(0f, 0f, 0f);
        Block_orange.transform.position = ground.transform.position + new Vector3(x_orange, 0.5f, z_orange);
        Block_red.transform.position = ground.transform.position + new Vector3(x_red, 0.5f, z_red);

        // Wall: reset position
        Wall.position = ground.transform.position + new Vector3(0, 0.5f, -8f);

        // Blocks: reset variables
        Block_orange.GetComponent<Block_col>().block_ok = false;
        Block_red.GetComponent<Block_col>().block_ok = false;
        Block_orange.GetComponent<Block_col>().block_fail = false;
        Block_red.GetComponent<Block_col>().block_fail = false;

        // Zones: reset variables
        orange_set = false;
        red_set = false;
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

        // Blocks and wall control
        var block_orange = Block_orange.GetComponent<Block_col>().block_ok;
        var block_red = Block_red.GetComponent<Block_col>().block_ok;
        var block_orange_fail = Block_orange.GetComponent<Block_col>().block_fail;
        var block_red_fail = Block_red.GetComponent<Block_col>().block_fail;

        if(block_orange_fail || block_red_fail) 
        {
            AddReward(-1f);
            AgentReset();
            Done();
        }

        if (block_orange && !orange_set)
        {
            orange_set = true;
            AddReward(0.2f);

            if (red_set)
            {
                AddReward(0.2f);
            }
        }

        if (block_red && !red_set)
        {
            red_set = true;
            AddReward(0.2f);

            if (orange_set)
            {
                AddReward(0.2f);
            }
        }

        if (block_orange && block_red)
        {
            Wall.position = ground.transform.position + new Vector3(0, -1f, -8f);
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
        AddReward(0.4f);
        AgentReset();
        Done(); 
    }

    void OnTriggerEnter(Collider col)
    {
        if (col.gameObject.CompareTag("green_zone"))
        {
            // Green zone touched
            ScoredAGoal();
        }
    }

    void Difficulty()
    {
        if (level == 1)
        {
            x_agent = 0f;
            z_agent = 0f;
            x_orange = -6.5f;
            z_orange = 6.5f;
            x_red = 6.5f;
            z_red = 6.5f;
         }  
        
        else if (level == 2)
        {
            x_agent = 0f;
            z_agent = 0f;
            x_orange = Random.Range(-11f,-2f);
            z_orange = Random.Range(2f, 6.5f);
            x_red = Random.Range(2f, 11f);
            z_red = Random.Range(2f, 6.5f);
        }

        else if (level == 3)
        {
            x_agent = 0f;
            z_agent = 0f;
            x_orange = Random.Range(-11f, -2f);
            z_orange = Random.Range(-4f, 6.5f);
            x_red = Random.Range(2f, 11f);
            z_red = Random.Range(-4f, 6.5f);
        }

        else if (level == 4)
        {
            x_agent = Random.Range(-11f, 11f);
            z_agent = Random.Range(-5f, 6.5f);
            x_orange = Random.Range(-11f, -2f);
            z_orange = Random.Range(-4f, 6.5f);
            x_red = Random.Range(2f, 11f);
            z_red = Random.Range(-4f, 6.5f);
        }

        else if (level == 5)
        {
            x_agent = Random.Range(-11f, 11f);
            z_agent = Random.Range(-5f, 6.5f);
            x_orange = Random.Range(-11f, 2f);
            z_orange = Random.Range(-4f, 6.5f);
            x_red = Random.Range(-2f, 11f);
            z_red = Random.Range(-4f, 6.5f);
        }

        else if (level == 6)
        {
            x_agent = Random.Range(-11f, 11f);
            z_agent = Random.Range(-5f, 6.5f);
            x_orange = Random.Range(-11f, 9f);
            z_orange = Random.Range(-4f, 6.5f);
            x_red = Random.Range(-9f, 11f);
            z_red = Random.Range(-4f, 6.5f);
        }
    }
}


