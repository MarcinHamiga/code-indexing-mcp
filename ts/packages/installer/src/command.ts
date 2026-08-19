import { Command, Option } from "commander";
import { ACCELERATOR_CHOICES } from "./accelerator.ts";
import { configureMain } from "./cli.ts";
import { uninstallMain } from "./uninstall.ts";
import { updateMain } from "./update.ts";

function collect(value: string, previous: string[]): string[] {
  return [...previous, value];
}

export const commandHandlers = {
  configure: configureMain,
  update: updateMain,
  uninstall: uninstallMain,
};

export async function main(argv: readonly string[] = process.argv.slice(2)): Promise<number> {
  const program = new Command();
  let returnCode = 0;
  program.name("code-indexing-mcp").exitOverride();
  program.configureOutput({
    writeErr: (text) => {
      process.stderr.write(text);
    },
  });

  program
    .command("configure")
    .description("Reconfigure this installation")
    .option("--install-dir <path>")
    .addOption(new Option("--accelerator <name>").choices([...ACCELERATOR_CHOICES]))
    .option("--harnesses <selection>")
    .option("--set <NAME=VALUE>", "set a managed setting", collect, [])
    .option("--unset <NAME>", "remove a managed setting", collect, [])
    .option("--bin-dir <path>")
    .option("--no-launcher")
    .option("--no-modify-path")
    .option("--no-tui")
    .option("--repair")
    .action(async (options) => {
      returnCode = await commandHandlers.configure({
        installDir: options.installDir,
        accelerator: options.accelerator,
        harnesses: options.harnesses,
        settings: options.set,
        unsets: options.unset,
        noTui: options.tui === false,
        binDir: options.binDir,
        noLauncher: options.launcher === false,
        noModifyPath: options.modifyPath === false,
        repair: options.repair === true,
      });
    });

  program
    .command("update")
    .description("Update this installation to the latest main")
    .option("--install-dir <path>")
    .option("--check")
    .option("--skip-accelerator")
    .addOption(new Option("--finalize").hideHelp())
    .addOption(new Option("--previous-sha <sha>").hideHelp())
    .action(async (options) => {
      returnCode = await commandHandlers.update({
        installDir: options.installDir,
        check: options.check === true,
        skipAccelerator: options.skipAccelerator === true,
        finalize: options.finalize === true,
        previousSha: options.previousSha,
      });
    });

  program
    .command("uninstall")
    .description("Remove client entries, skills, and the launcher")
    .option("--install-dir <path>")
    .option("--harnesses <selection>")
    .option("--bin-dir <path>")
    .option("--keep-launcher")
    .option("--keep-path")
    .option("--purge")
    .option("--remove-checkout")
    .option("--yes")
    .action(async (options) => {
      returnCode = await commandHandlers.uninstall({
        installDir: options.installDir,
        harnessesSelection: options.harnesses,
        binDir: options.binDir,
        keepLauncher: options.keepLauncher === true,
        keepPath: options.keepPath === true,
        purge: options.purge === true,
        removeCheckout: options.removeCheckout === true,
        assumeYes: options.yes === true,
      });
    });

  try {
    await program.parseAsync(["node", "code-indexing-mcp", ...argv], { from: "node" });
    return returnCode;
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "commander.helpDisplayed") {
      return 0;
    }
    if (error instanceof Error && "code" in error && String(error.code).startsWith("commander.")) {
      return 1;
    }
    throw error;
  }
}

if (import.meta.main) {
  void main().then(
    (code) => {
      process.exitCode = code;
    },
    (error: unknown) => {
      console.error(error);
      process.exitCode = 1;
    },
  );
}
