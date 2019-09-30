本项目目录：
-------result
	-------result.txt		#结果存储文档

-------data
	-------xxx.txt		#多个测试或训练txt文档
	..................

-------Bayes.py			#源码
	
-------READ.txt

注意事项：
Windows环境(推荐win10)
python环境要求：python3且装有numpy库。
可添加到pycharm，Anaconda， Geany等能运行python的工程中进行调试。
亦可直接在命令行下执行。

PS:在.py文件中默认是以数据量最大的对应的两个txt作为训练数据。可通过修改110行和127行对其他训练数据进行测试。

重要！PS：必须保证data文件目录与Bayes.py文件在同一目录下，同时data其中的文件保证完整性。