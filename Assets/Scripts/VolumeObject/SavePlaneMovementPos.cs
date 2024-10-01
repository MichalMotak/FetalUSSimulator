// using UnityEngine;
// using UnityEngine.UI;
// using System;
// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine.SceneManagement;
// #if UNITY_EDITOR
// using UnityEditor;
// using UnityEditorInternal;
// using System.Text;
// using System.IO;
// using System.Threading.Tasks;

// namespace UnityVolumeRendering
// {
// 	public class SavePlaneMovementPos : MonoBehaviour
// 	{		
// 		public List<string[]> rowData = new List<string[]>();
// 		private string filePath = "AcquiredData/Poses2/pos_unity.csv";
// 		private string imagePath = "AcquiredData";


// 		// Całość
		
// 		// private float initial_x = -0.04f, initial_y = -0.001f, initial_z = -0.19f; // -0.19
// 		// private float final_x = 0.042f, final_y = 0.001f, final_z = 0.19f; // 0.19
// 		// private float update_x = 0.009f, update_y = 0.001f, update_z = 0.003f;
// 		// private int waitForMilliSeconds = 250;


// 		// Zapisałem to w folderze Poses1 ---------------------------------------------------
// 		// private float initial_x = -0.04f, initial_y = -0.001f, initial_z = -0.19f; // -0.19
// 		// private float final_x = 0.00f, final_y = 0.00f, final_z = 0.00f; // 0.19
// 		// private float update_x = 0.009f, update_y = 0.001f, update_z = 0.003f;
// 		// private int waitForMilliSeconds = 250;
// 		// ---------------------------------------------------------------------------

// 		// Zapisałem to w folderze Poses2 ---------------------------------------------------

// 		private float initial_x = 0.00f, initial_y = -0.001f, initial_z = 0.00f; // -0.19
// 		private float final_x = 0.042f, final_y = 0.00f, final_z = 0.19f; // 0.19
// 		private float update_x = 0.009f, update_y = 0.001f, update_z = 0.003f;
// 		private int waitForMilliSeconds = 250;




// 		public string[] rowDataTemp = new string[7];

// 		// private float y_offset_up = 21.0f;
// 		private float y_offset_up = 27.0f;
// 		private float y_offset_down = 100.0f;

// 		public int screenshotIndex = 0;

// 		public delegate void FunctionBDelegate();
// 		public static event FunctionBDelegate OnFunctionB;


// 	    // Start is called before the first frame update
// 	    void Start()
// 	    {
// 	    	// Creating First row of titles manually
// 	    	string[] StartrowDataTemp = new string[7];
// 	        StartrowDataTemp[0] = "pos_x";
// 	        StartrowDataTemp[1] = "pos_y";
// 	        StartrowDataTemp[2] = "pos_z";
// 	        StartrowDataTemp[3] = "rot_x";
// 	        StartrowDataTemp[4] = "rot_y";
// 	        StartrowDataTemp[5] = "rot_z";
// 	        StartrowDataTemp[6] = "id";

// 	        rowData.Add(StartrowDataTemp);

// 	    	EditorApplication.ExecuteMenuItem("Volume Rendering/Slice acquisition");

// 	        SetPose();

// 			// StartCoroutine(SetPose());

// 			// IEnumerator ExecuteFunctionA()
// 			// {
// 			// 	Debug.Log("Function A is running");
// 			// 	yield return new WaitForSeconds(2); // Simulate work with delay
// 			// 	Debug.Log("Function A completed");

// 			// 	FunctionAComplete?.Invoke();
// 			// }


// 	    }

// 	    public async Task MyAsyncMethod()
// 		{
// 		    await Task.Delay(waitForMilliSeconds);
// 		}

// 	    // Update is called once per frame
// 	   	public async void SetPose()
// 	    {
// 	    	var activeWindow = EditorWindow.focusedWindow;
// 			var vec2Position = activeWindow.position.position;                     
//             var sizeX = activeWindow.position.width;
//             var sizeY = activeWindow.position.height;
//             var sizeX_plane = sizeX - sizeX * 0.5f;
//             var sizeY_plane = sizeY - y_offset_down;
//             // var sizeX_plane = sizeX;
//             // var sizeY_plane = sizeY;


// 	    	for(float idx_x = initial_x; idx_x < final_x; idx_x += update_x)
// 	    	{
// 	    		for(float idx_y = initial_y; idx_y < final_y; idx_y += update_y)
// 	    		{
// 			    	for(float idx_z = initial_z; idx_z < final_z; idx_z += update_z)
// 			    	{
// 			    		// Vector3 temppos = new Vector3(idx_x, idx_y, idx_z);
// 			    		Vector3 temppos = new Vector3(idx_x, idx_y, idx_z);
// 			    		transform.localPosition = temppos;
// 			    		Quaternion temprot = transform.localRotation;

// 		    			// You can add up the values in as many cells as you want
// 				        string[] rowDataTemp = new string[7];
// 				        rowDataTemp[0] = temppos.x.ToString();
// 					    rowDataTemp[1] = temppos.y.ToString();
// 					    rowDataTemp[2] = temppos.z.ToString();
// 					    rowDataTemp[3] = temprot.x.ToString();
// 					    rowDataTemp[4] = temprot.y.ToString();
// 					    rowDataTemp[5] = temprot.z.ToString();
// 						rowDataTemp[6] = screenshotIndex.ToString();

// 				        rowData.Add(rowDataTemp);

// 				        // var colors = InternalEditorUtility.ReadScreenPixel(new Vector2(vec2Position.x, (vec2Position.y + y_offset_up)), (int)sizeX_plane, (int)sizeY_plane);
// 	                    // var result = new Texture2D((int)sizeX_plane, (int)sizeY_plane, TextureFormat.RGB24, false);
// 	                    // result.SetPixels(colors);
// 	                    // var bytes = result.EncodeToPNG();
// 	                    // // Object.DestroyImmediate(result);
// 	                    // DestroyImmediate(result);
// 	                    // File.WriteAllBytes(Path.Combine(imagePath, "plane" + screenshotIndex + ".png"), bytes);
// 	                    screenshotIndex++;
// 	                    // AssetDatabase.Refresh();
// 	                    // Debug.Log("New Screenshot taken");

// 						Debug.Log($" ind {screenshotIndex}");
// 						Debug.Log($" idx_x {idx_x}");
// 						Debug.Log($" idx_y {idx_y}");
// 						Debug.Log($" idx_z {idx_z}");
// 						Debug.Log($" rowDataTemp {rowDataTemp}");

// 				        await MyAsyncMethod(); 
// 						OnFunctionB.Invoke();


// 					}
// 				}
// 	    	}

// 	        string[][] output = new string[rowData.Count][];

// 	        for(int i = 0; i < output.Length; i++)
// 	        {
// 	            output[i] = rowData[i];
// 	        }

//         	int length = output.GetLength(0);
//         	string delimiter = ",";
//         	StringBuilder sb = new StringBuilder();
        
// 	        for (int index = 0; index < length; index++)
// 	        {
// 	            sb.AppendLine(string.Join(delimiter, output[index]));
// 	        }

// 	        StreamWriter outStream = System.IO.File.CreateText(filePath);
// 	        outStream.WriteLine(sb);
// 	        outStream.Close();
// 			Debug.Log("Csv file saved");
// 			EditorApplication.isPlaying = false;	




// 	    }
// 	}
// }

// #endif




// // using UnityEngine;
// // using UnityEngine.UI;
// // using System;
// // using System.Collections;
// // using System.Collections.Generic;
// // using UnityEngine.SceneManagement;
// // using UnityEditor;
// // using UnityEditorInternal;
// // using System.Text;
// // using System.IO;
// // using System.Threading.Tasks;

// // namespace UnityVolumeRendering
// // {
// // 	public class PlaneMovementPos : MonoBehaviour
// // 	{		
// // 		private List<string[]> rowData = new List<string[]>();
// // 		private string filePath = "AcquiredData/Poses/pos_unity.csv";
// // 		private float initial_x = -0.04f, initial_y = -0.001f, initial_z = -0.19f; // -0.19
// // 		private float final_x = 0.042f, final_y = 0.001f, final_z = 0.19f; // 0.19
// // 		private float update_x = 0.009f, update_y = 0.001f, update_z = 0.003f;
// // 		private int waitForMilliSeconds = 100;

// // 		private int screenshotIndex = 3810;
// // 		private float bgWidth;
// // 		private Rect bgRect;
// // 		private Vector2 screenPosition;

// // 	    // Start is called before the first frame update
// // 	    void Start()
// // 	    {
// // 	    	// Creating First row of titles manually
// // 	    	string[] rowDataTemp = new string[6];
// // 	        rowDataTemp[0] = "pos_x";
// // 	        rowDataTemp[1] = "pos_y";
// // 	        rowDataTemp[2] = "pos_z";
// // 	        rowDataTemp[3] = "rot_x";
// // 	        rowDataTemp[4] = "rot_y";
// // 	        rowDataTemp[5] = "rot_z";
// // 	        rowData.Add(rowDataTemp);

// //             bgWidth = Mathf.Min(800.0f - 20.0f, (500.0f - 50.0f) * 2.0f);
// //             bgRect = new Rect(0.0f, 0.0f, (float)(bgWidth*0.5), (float)(bgWidth*0.5));

// //             screenPosition.x = bgRect.x;
// //             screenPosition.y = bgRect.y + 66.0f;

// // 	        SetPose();
// // 	    }

// // 	    public async Task MyAsyncMethod()
// // 		{
// // 		    await Task.Delay(waitForMilliSeconds);
// // 		}

// // 	    // Update is called once per frame
// // 	   	public async void SetPose()
// // 	    {
// // 	    	for(float idx_x = initial_x; idx_x < final_x; idx_x += update_x)
// // 	    	{
// // 	    		// for(float idx_y = initial_y; idx_y < final_y; idx_y += update_y)
// // 	    		// {
// // 			    	for(float idx_z = initial_z; idx_z < final_z; idx_z += update_z)
// // 			    	{
// // 			    		// Vector3 temppos = new Vector3(idx_x, idx_y, idx_z);
// // 			    		Vector3 temppos = new Vector3(idx_x, 0f, idx_z);
// // 			    		transform.position = temppos;
// // 			    		Quaternion temprot = transform.rotation;

// // 		    			// You can add up the values in as many cells as you want
// // 				        string[] rowDataTemp = new string[6];
// // 				        rowDataTemp[0] = temppos.x.ToString();
// // 					    rowDataTemp[1] = temppos.y.ToString();
// // 					    rowDataTemp[2] = temppos.z.ToString();
// // 					    rowDataTemp[3] = temprot.x.ToString();
// // 					    rowDataTemp[4] = temprot.y.ToString();
// // 					    rowDataTemp[5] = temprot.z.ToString();
// // 				        rowData.Add(rowDataTemp);

// // 				        var colors = InternalEditorUtility.ReadScreenPixel(screenPosition, (int)(bgWidth*0.5), (int)(bgWidth*0.5f));
// // 	                    var result = new Texture2D((int)(bgWidth*0.5), (int)(bgWidth*0.5f), TextureFormat.RGB24, false);
// // 	                    result.SetPixels(colors);
// // 	                    var bytes = result.EncodeToPNG();
// // 	                    // Object.DestroyImmediate(result);
// // 	                    DestroyImmediate(result);
// // 	                    File.WriteAllBytes(Path.Combine("AcquiredData", "plane" + screenshotIndex + ".png"), bytes);
// // 	                    screenshotIndex++;
// // 	                    AssetDatabase.Refresh();
// // 	                    Debug.Log("New Screenshot taken");


// // 						// // Take a screenshot.
// // 						// texture.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
// // 						// texture.Apply();
						  
// // 						// // Save the screenshot as a PNG file.
// // 						// ES3.SaveImage(texture, "screenshot.png");

// // 						// SaveTexture();

// // 				        await MyAsyncMethod();
// // 					}
// // 				// }
// // 	    	}

// // 	        string[][] output = new string[rowData.Count][];

// // 	        for(int i = 0; i < output.Length; i++)
// // 	        {
// // 	            output[i] = rowData[i];
// // 	        }

// //         	int length = output.GetLength(0);
// //         	string delimiter = ",";
// //         	StringBuilder sb = new StringBuilder();
        
// // 	        for (int index = 0; index < length; index++)
// // 	        {
// // 	            sb.AppendLine(string.Join(delimiter, output[index]));
// // 	        }

// // 	        StreamWriter outStream = System.IO.File.CreateText(filePath);
// // 	        outStream.WriteLine(sb);
// // 	        outStream.Close();
// // 	    }
// // 	}
// // }